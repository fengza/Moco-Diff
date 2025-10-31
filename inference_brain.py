from typing import List, Tuple, Optional
import os
import math
from argparse import ArgumentParser, Namespace
import sys
import time
import random
import torchio as tio
import numpy as np

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
import fid
import pyiqa
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.cldm import ControlLDM
from model.cond_fn import MSEGuidance
from utils.file import load_file_list
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts,list_image_files_natural
import SimpleITK as sitk
from utils.image import center_crop_arr, augment, random_crop_arr,center_crop_arr_2d
from utils.metrics import calculate_psnr_pt, LPIPS, calculate_ssim_pt
from dataset import motion_sim
import imageio
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def load_nii(nii_path):
    try:
        image = sitk.ReadImage(nii_path)
    except:
        assert False, f"failed to load image {nii_path}"
    image_array = sitk.GetArrayFromImage(image)
    
    return image_array

def dcm_to_nii(folder_path):
    try:
        reader = sitk.ImageSeriesReader()
        dicomFiles = reader.GetGDCMSeriesFileNames(folder_path)
        dicom_itk = sitk.ReadImage(dicomFiles)
        # img_arr = sitk.GetArrayFromImage(dicom_itk)
        # file_name = folder_path.split('/')[-1]
        #file_name = file_name.rjust(3,'0') + '_0000.nii.gz'
        save_path = os.path.join('/public_bme/data/lifeng/data/moco/test/CCBD/ccbd_1','gt')
        save_file = save_path + '.nii.gz'
        sitk.WriteImage(dicom_itk, save_file)
    except:
        return None

@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    cond_fn: Optional[MSEGuidance],
    tiled: bool,
    tile_size: int,
    tile_stride: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply DiffBIR model on a list of low-quality images.
    
    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(np.stack(control_imgs), dtype=torch.float32, device=model.device).clamp_(0, 1)
    #control = einops.rearrange(control, "n c h w -> n c h w").contiguous()
    control = einops.rearrange(control, "n d h w c -> n d c h w").contiguous()  #(12,3,512,512)
    
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    model.control_scales = [strength] * 13
    
    if cond_fn is not None:
        cond_fn.load_target(2 * control - 1)
    
    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)   #(n_samples, 4, 64, 64)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if not tiled:
        samples = sampler.sample(
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    else:
        samples = sampler.sample_with_mixdiff(
            tile_size=tile_size, tile_stride=tile_stride,
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    x_samples = samples.clamp(0, 1)
    # x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    # control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c")).cpu().numpy()
    control = (einops.rearrange(control, "b c h w -> b h w c")).cpu().numpy()
    
    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]
    
    return preds, stage1_preds


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt", required=False, type=str, default='/public_bme/data/lifeng/code/moco/MoCo-Diff/weights/lightning_logs/version_1412213/checkpoints/step=649.ckpt', help="full checkpoint path") #/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints/cldm.ckpt
    parser.add_argument("--config", required=False, type=str, default='/public_bme/data/lifeng/code/moco/MoCo-Diff/configs/model/cldm.yaml', help="model config path")
    parser.add_argument("--reload_swinir", action="store_true", default=True)
    parser.add_argument("--swinir_ckpt", type=str, default="/public_bme/data/lifeng/code/moco/MoCo-Diff/checkpoints/lightning_logs/version_1411889/checkpoints/step=389.ckpt")
    #/public_bme/data/lifeng/code/moco/TS_BHIR/checkpoints/lightning_logs/version_1411889/checkpoints/step=499.ckpt
    parser.add_argument("--input", type=str, default='/public_bme/data/lifeng/code/moco/DiffBIR-main/inputs/demo/T1_brain', required=False)
    parser.add_argument("--steps", required=False, type=int, default=50) #50
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--repeat_times", type=int, default=3)
    parser.add_argument("--disable_preprocess_model", action="store_true",default=False)
    
    # patch-based sampling
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    
    # latent image guidance
    parser.add_argument("--use_guidance", action="store_true", default=True)
    parser.add_argument("--g_scale", type=float, default=150) #300提升，200会提升一点lpips会降低，150是本身的值, 100的不行值太低
    parser.add_argument("--g_t_start", type=int, default=200)
    parser.add_argument("--g_t_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=5)
    
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str, required=False, default='/public_bme/data/lifeng/data/moco/hcp_3d/test')
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
     
    parser.add_argument("--seed", type=int, default=123)  #231
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    
    return parser.parse_args()

def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device

def get_data(file_list, out_size = 512):
    paths = load_file_list(file_list)#[15:25]
    #paths = ['/public_bme/data/lifeng/data/moco/test/CCBD/ccbd_1/gt_1.nii.gz']
    #num_samples = len(paths)
    pbar = tqdm(range(len(paths)))
    
    target_all = []
    #source_all = []
    #ids = 0
    for index in pbar: #range(num_samples): #num_samples
        gt_path = paths[index]#.replace('T1w','T2w')
        success = False
        try:
            pil_img_3D = load_nii(gt_path)
            success = True

            #transformed_img_3D = load_nii(gt_path.replace('gt', 'lq'))
        except:
            continue
        assert success, f"failed to load image {gt_path}"

        # # ------------------------ generate lq image ------------------------ #
            
        ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[20, 20],  #[0,25]
                                            corruption_scheme='piecewise_constant',#'piecewise_transient',
                                            n_seg=8)
        transformed_img_3D = ms_layer.layer_op(pil_img_3D)

        lq_nii = np.array(transformed_img_3D)
        lq_nii = sitk.GetImageFromArray(lq_nii)
        sitk.WriteImage(lq_nii,f'/public_bme/data/lifeng/data/moco/hcp_simulation/train_set/test/lq_3d.nii.gz')

def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    
    args.device = check_device(args.device)
    
    model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))   #创建模型
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)     #模型的预训练权重
    # reload preprocess model if specified
    if args.reload_swinir:
        if not hasattr(model, "preprocess_model"):
            raise ValueError(f"model don't have a preprocess model.")
        print(f"reload swinir model from {args.swinir_ckpt}")
        load_state_dict(model.preprocess_model, torch.load(args.swinir_ckpt, map_location="cpu"), strict=True)
    model.freeze()
    model.to(args.device)
    
    lpips_metric = LPIPS(net="alex")
    
    #target_all, source_all = get_data('/home_data/home/lifeng2023/data/diffbir_data/val.list')
    # num_samples = target_all.shape[0]

    #----- Test -----#
    pil_img_3D = load_nii('/public_bme/data/lifeng/data/hcp/352738/T1w_hires.nii.gz')
    ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[25, 25],
                                        corruption_scheme='gaussian',#'piecewise_transient',
                                        n_seg=8)
    lq_img_3D = ms_layer.layer_op(pil_img_3D)

    length = pil_img_3D.shape[0]
    median = round(length / 2)

    target_all = []
    source_all = []
    #for i in index_A:
    #for i in range(median-40, median+50):
    for i in range(100, 120):
        img_gt = pil_img_3D[i-2:i+4]
        lq1_img = lq_img_3D[i-2:i+4]
        #lq2_img = lq2_img_3D[i-2:i+4]
        
        img_gt = center_crop_arr(img_gt, 512, dim = 3)
        lq1_img = center_crop_arr(lq1_img, 512, dim = 3)
        #lq2_img = center_crop_arr(lq2_img, 512)

        gt = img_gt.astype(np.float32)#(img_gt * 2 - 1).astype(np.float32)

        #[0, 1]
        x = lq1_img.astype(np.float32)

        target_all.append(gt)
        source_all.append(x)
    
    target_all = np.array(target_all)
    source_all = np.array(source_all)

    pbar = tqdm(range(target_all.shape[0]))
    #assert target_all.shape == source_all.shape, "target and source have different size"
    psnr_all = []
    ssim_all = []
    lpips_all = []
    fid_all = []
    niqe_all = []
    musiq_all = []

    psnr_all_lq = []
    ssim_all_lq = []
    lpips_all_lq = []
    fid_all_lq = []
    niqe_all_lq = []
    musiq_all_lq = []
    # num_samples = target_all.shape[0]
    # for index in range(num_samples):
    pred_list = []
    gt_list = []
    lq_list = []

    for index in pbar:      
        x = source_all[index] 
        gt= target_all[index] #2d
        #gt = (einops.rearrange(gt, "c h w -> h w c") * 255).clip(0, 255).astype(np.uint8)     
        #gt = einops.rearrange(gt, "c h w -> h w c")      
        # initialize latent image guidance
        if args.use_guidance:
            cond_fn = MSEGuidance(
                scale=args.g_scale, t_start=args.g_t_start, t_stop=args.g_t_stop,
                space=args.g_space, repeat=args.g_repeat
            )
        else:
            cond_fn = None

        lq = np.stack((x[2], x[2], x[2]), axis=0)  
        lq_resized = np.stack((x, x, x), axis=-1)  
        gt = np.stack((gt[2], gt[2], gt[2]), axis=0)  

        """Normalize img to [0,1]
        """
        max_value, min_value = lq_resized.max(), lq_resized.min() 
        lq_resized = (lq_resized - min_value) / (max_value - min_value)

        max_value, min_value = gt.max(), gt.min() 
        gt = (gt - min_value) / (max_value - min_value)

        max_value, min_value = lq.max(), lq.min() 
        lq = (lq - min_value) / (max_value - min_value)

        gt_list.append(gt[0])
        lq_list.append(lq[0])

        x = lq_resized 
        preds, stage1_preds = process(
            model, [x]*12, steps=args.steps,  ##(12,6,512,512,3)
            strength=1,
            color_fix_type=args.color_fix_type,
            disable_preprocess_model=args.disable_preprocess_model,
            cond_fn=cond_fn,
            tiled=args.tiled, tile_size=args.tile_size, tile_stride=args.tile_stride
        )
        pred, stage1_pred = preds[0], stage1_preds[0]
        pred = einops.rearrange(pred, "h w c -> c h w")

        pred_list.append(pred[0])
       
        #save img
        #pred_img = pred#.clamp(0, 1)#.cpu().numpy()
        pred_img = pred.transpose(1,2,0)  #
        gt_img = gt.transpose(1,2,0)  
        lq_img = lq.transpose(1,2,0)  
        
        pred_img = (pred_img * 255).clip(0, 255).astype(np.uint8)
        gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)
        lq_img = (lq_img * 255).clip(0, 255).astype(np.uint8)

        psnr = calculate_psnr_pt(pred, gt, crop_border=0).mean()  #27.2485
        pred_ = torch.tensor(pred , dtype=torch.float32)  #tensor(3,512,512)
        gt_ = torch.tensor(gt, dtype=torch.float32)      #tensor(3,512,512)
        lpips = lpips_metric(pred_, gt_, normalize=True).mean()  #0.2679
        pred_2 = np.array([pred])   #(1,3,512,512)
        gt_2 = np.array([gt])       #(1,3,512,512)
        ssim = calculate_ssim_pt(pred_2, gt_2, crop_border=0).mean()  #0.7713
        FID = fid.calculate_fid(gt_2.transpose(0,2,3,1), pred_2.transpose(0,2,3,1), False, 1)
        musiq_metric = pyiqa.create_metric('musiq',device='cuda')
        MUSIQ = musiq_metric(pred_.unsqueeze(0)).detach().cpu().numpy()[0][0]
        niqe_metric = pyiqa.create_metric('niqe',device='cuda')
        NIQE = niqe_metric(pred_.unsqueeze(0)).detach().cpu().numpy()

        psnr_all.append(np.float32(psnr))
        lpips_all.append(np.float32(lpips))
        ssim_all.append(np.float32(ssim))
        fid_all.append(np.float32(FID))
        niqe_all.append(np.float32(NIQE))
        musiq_all.append(np.float32(MUSIQ))
        #nrmse_all.append(np.float32(NRMSE))

        #calculate lq metrics
        psnr_lq = calculate_psnr_pt(lq, gt, crop_border=0).mean()  #27.2485
        lq_ = torch.tensor(lq , dtype=torch.float32)  #tensor(3,512,512)
        #gt_ = torch.tensor(gt, dtype=torch.float32)      #tensor(3,512,512)
        lpips_lq = lpips_metric(lq_, gt_, normalize=True).mean()  #0.2679
        lq_2 = np.array([lq])   #(1,3,512,512)
        #gt_2 = np.array([gt])       #(1,3,512,512)
        ssim_lq = calculate_ssim_pt(lq_2, gt_2, crop_border=0).mean()  #0.7713
        FID_lq = fid.calculate_fid(gt_2.transpose(0,2,3,1), lq_2.transpose(0,2,3,1), False, 1)
        musiq_metric = pyiqa.create_metric('musiq',device='cuda')
        MUSIQ_lq = musiq_metric(lq_.unsqueeze(0)).detach().cpu().numpy()[0][0]
        niqe_metric = pyiqa.create_metric('niqe',device='cuda')
        NIQE_lq = niqe_metric(lq_.unsqueeze(0)).detach().cpu().numpy()
        #NRMSE_lq = calculate_nrmse(pred, lq)

        psnr_all_lq.append(np.float32(psnr_lq))
        lpips_all_lq.append(np.float32(lpips_lq))
        ssim_all_lq.append(np.float32(ssim_lq))
        fid_all_lq.append(np.float32(FID_lq))
        niqe_all_lq.append(np.float32(NIQE_lq))
        musiq_all_lq.append(np.float32(MUSIQ_lq))
        #nrmse_all_lq.append(np.float32(NRMSE_lq))

        path = os.path.join(args.output, f'pred_{index}_psnr({psnr}_ssim({ssim})_lpips({lpips}.png')
        path_gt = os.path.join(args.output, f'gt_{index}.png')
        path_lq = os.path.join(args.output, f'lq_{index}_psnr({psnr_lq}_ssim({ssim_lq})_lpips({lpips_lq}.png')
        Image.fromarray(pred_img).save(path)
        Image.fromarray(gt_img).save(path_gt)
        Image.fromarray(lq_img).save(path_lq)
        #loss = get_loss(pred, gt)
    pred_arr_3d = np.array(pred_list)
    gt_arr_3d = np.array(gt_list)
    lq_arr_3d = np.array(lq_list)

    pred_img_3d = sitk.GetImageFromArray(pred_arr_3d)
    sitk.WriteImage(pred_img_3d, os.path.join(args.output, 'pred_3d_352738.nii.gz'))
    gt_img_3d = sitk.GetImageFromArray(gt_arr_3d)
    sitk.WriteImage(gt_img_3d, os.path.join(args.output, 'gt_3d_352738.nii.gz'))
    lq_img_3d = sitk.GetImageFromArray(lq_arr_3d)
    sitk.WriteImage(lq_img_3d, os.path.join(args.output, 'lq_3d_352738.nii.gz'))

    print("psnr:",np.mean(psnr_all),np.std(psnr_all))
    print("ssim:",np.mean(ssim_all),np.std(ssim_all))
    print("lpips:",np.mean(lpips_all),np.std(lpips_all))
    print("fid:",np.mean(fid_all),np.std(fid_all))
    print("musiq:",np.mean(musiq_all),np.std(musiq_all))
    print("niqe:",np.mean(niqe_all),np.std(niqe_all))

    print("psnr_lq:",np.mean(psnr_all_lq),np.std(psnr_all_lq))
    print("ssim_lq:",np.mean(ssim_all_lq),np.std(ssim_all_lq))
    print("lpips_lq:",np.mean(lpips_all_lq),np.std(lpips_all_lq))
    print("fid_lq:",np.mean(fid_all_lq),np.std(fid_all_lq))
    print("musiq_lq:",np.mean(musiq_all_lq),np.std(musiq_all_lq))
    print("niqe_lq:",np.mean(niqe_all_lq),np.std(niqe_all_lq))


if __name__ == "__main__":
    main()
    #print('The psnr is {}, lpips is {}, ssim is {}'.format(pnsr, lpips, ssim))
    #get_data('/public_bme/data/lifeng/data/val_hcp.list')