import sys
#sys.path.append(".")
sys.path.append("/public_bme/data/lifeng/code/moco/TS_BHIR")
import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import einops
from utils.image import auto_resize, pad
from utils.common import load_state_dict, instantiate_from_config
from utils.file import list_image_files, get_file_name_parts,list_image_files_natural,load_file_list
from utils.metrics import calculate_psnr_pt, LPIPS, calculate_ssim_pt
from utils.image import center_crop_arr
from utils.common import frozen_module
import fid
from inference_brain import get_data
import pyiqa
import statistics
import skimage.measure
import SimpleITK as sitk
from dataset import motion_sim
import random
import cv2
import imageio
import matplotlib.pyplot as plt
from dataset.motionbrain_revised import normalize_mri_volume

def calculate_nrmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets)**2)) / (np.max(targets) - np.min(targets))

def load_nii(nii_path):
    try:
        image = sitk.ReadImage(nii_path)
    except:
        assert False, f"failed to load image {nii_path}"
    image_array = sitk.GetArrayFromImage(image)
    
    return image_array

def normalize_mri_volume(volume):
    """
    Normalize the MRI 3D volume by clipping the intensity values to the 0.3%-99.7% range
    and setting values above 99.7% to the maximum value in the clipped range.

    Parameters:
    volume (numpy.ndarray): The input 3D MRI volume.

    Returns:
    numpy.ndarray: The normalized 3D MRI volume.
    """
    # Compute the 0.3% and 99.7% intensity values
    lower_bound = np.percentile(volume, 0.3)
    upper_bound = np.percentile(volume, 99.7)
    
    # # Clip the intensity values to the 0.3%-99.7% range
    # clipped_volume = np.clip(volume, lower_bound, upper_bound)
    
    # Set values above 99.7% to the maximum value in the clipped range
    volume[volume > upper_bound] = upper_bound
    volume[volume < lower_bound] = lower_bound
    
    return volume#.astype("float32")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,  default='/public_bme/data/lifeng/code/moco/TS_BHIR/configs/model/swinir.yaml', required=False)
    #parser.add_argument("--ckpt", type=str, default="/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints/lightning_logs/version_1285486/checkpoints/step=999.ckpt", required=False)#/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints
    parser.add_argument("--ckpt", type=str, default="/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/swinir_512_axis/checkpoints_new/lightning_logs/version_2977616/checkpoints/step=39.ckpt", required=False) #249.ckpt best
    #/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints/lightning_logs/version_1386969/checkpoints/step=5699.ckpt
    #/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints/lightning_logs/version_1410949/checkpoints/step=699.ckpt
    #/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints_imgrouting/lightning_logs/version_2894049/checkpoints/step=1303.ckpt
    #/public_bme/data/lifeng/code/moco/TS_BHIR/checkpoints/lightning_logs/version_1411889/checkpoints/step=499.ckpt # 249.ckpt best
    #parser.add_argument("--file_list", type=str, default="/public_bme/data/lifeng/data/val_hcp.list", required=False)
    parser.add_argument("--file_list", type=str, default="/home_data/home/lifeng2023/data/diffbir_data/val.list", required=False)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=512) #512
    parser.add_argument("--crop_type", type=str, default="none")    
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--resize_back", action="store_true")
    parser.add_argument("--output", type=str,  default="/public_bme2/bme-wangqian2/lifeng2023/data/restormer/hcp_slice/0/t1", required=False)
    parser.add_argument("--skip_if_exist", action="store_true")
    #parser.add_argument("--seed", type=int, default=123) #231,本来训练验证模型的种子
    return parser.parse_args()


def sim_motion(pil_img_3D):
    corruption_scheme = ['piecewise_constant','gaussion', 'piecewise_transient'] #
    selected_corruption_scheme = np.random.choice(corruption_scheme)
    
    ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[15, 30],
                                        corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                        n_seg=8)
    lq_img_3D = ms_layer.layer_op(pil_img_3D)

    return lq_img_3D


@torch.no_grad()
def main():
    args = parse_args()
    #pl.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model: pl.LightningModule = instantiate_from_config(OmegaConf.load(args.config))
    a = torch.load(args.ckpt, map_location="cpu")
    load_state_dict(model, a, strict=True)
    model.freeze()
    #frozen_module(model)
    model.to(device)
    psnr_all = []
    ssim_all = []
    lpips_all = []
    fid_all = []
    niqe_all = []
    musiq_all = []
    nrmse_all = []

    psnr_all_lq = []
    psnr_all_lq_bef = []
    ssim_all_lq = []
    lpips_all_lq = []
    fid_all_lq = []
    niqe_all_lq = []
    musiq_all_lq = []
    nrmse_all_lq = []

    pred_list = []
    gt_list = []
    lq_list = []
    #assert os.path.isdir(args.input)

    #target_all, source_all = get_data(args.file_list) #list, [512,512,3]*n*80

    # target_all = np.load('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_1530_hq.npy')
    # source_all = np.load('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_1530_lq.npy')
    
    # pil_img_3D = load_nii('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_intensity/7T/sub-02_gt.nii.gz')
    # transformed_img_3D = load_nii('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_intensity/7T/sub-02_lq.nii.gz')
    root_dir = '/public_bme/data/xhl/multi-degree/hcp_slice/0/t1/gt'
    paths = os.listdir(root_dir)
    #length = pil_img_3D.shape[0]
    #length = target_all.shape[0]

    # paths = load_file_list('/public_bme/data/lifeng/data/val_hcp.list')
    for index in range(len(paths)):
    #for index in range(length):
        #index += 189
        gt_path = os.path.join(root_dir, paths[index])
        lq_path = gt_path.replace('gt','lq')
        img_gt = imageio.imread(gt_path, pilmode='L')
        lq_single = imageio.imread(lq_path, pilmode='L')
        # try:
        #     pil_img_3D = load_nii(gt_path)
            
        # except:
        #     continue

        # pil_img_3D = center_crop_arr(pil_img_3D, args.image_size)
        # transformed_img_3D = sim_motion(pil_img_3D)
        # transformed_img_3D = normalize_mri_volume(transformed_img_3D)
        # max_value, min_value = transformed_img_3D.max(), transformed_img_3D.min() 
        # transformed_img_3D = (transformed_img_3D - min_value) / (max_value - min_value)

        # pil_img_3D = normalize_mri_volume(pil_img_3D)
        # max_value, min_value = pil_img_3D.max(), pil_img_3D.min() 
        # pil_img_3D = (pil_img_3D - min_value) / (max_value - min_value)

        # from utils import util_image
   
        # pil_img_3D = util_image.imread('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/gt_0_150.png',
        #                         chn = 'rgb', dtype='float32')
        # lq_img_3D = util_image.imread('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/lq_0_150_psnr(79.9731978784669_ssim(0.9999999704552437)_lpips(1.4173985540821832e-08.png',
        #                         chn = 'rgb', dtype='float32')

        # max_value, min_value = lq_img_3D.max(), lq_img_3D.min() 
        # lq_img_3D = (lq_img_3D - min_value) / (max_value - min_value)

        # max_value, min_value = pil_img_3D.max(), pil_img_3D.min() 
        # pil_img_3D = (pil_img_3D - min_value) / (max_value - min_value)

    # pil_img_3D_selected = sitk.GetImageFromArray(pil_img_3D[10:195])
    # sitk.WriteImage(pil_img_3D_selected, os.path.join(args.output, 'gt_s1_3d_352738_ori.nii.gz'))

    # lq_img_3D_selected = sitk.GetImageFromArray(lq_img_3D[10:195])
    # sitk.WriteImage(lq_img_3D_selected, os.path.join(args.output, 'lq_s1_3d_352738_ori.nii.gz'))
    
    #pbar = tqdm(list_image_files_natural(args.input, follow_links=True))

    #----- Test -----#
    # pil_img_3D = load_nii('/public_bme/data/lifeng/data/moco/openneuro/sub-000175/anat/sub-000175_acq-standard_T1w.nii.gz')
    # lq_img_3D = load_nii('/public_bme/data/lifeng/data/moco/openneuro/sub-000175/anat/sub-000175_acq-headmotion1_T1w.nii.gz')
    ##index_A = random.randint(130, 215) #[179, 180] #random.sample(range(168, 188), 2)
    # target_all = []
    # source_all = []
    # for i in range(138,195):
    #     img_gt = pil_img_3D[i-2:i+4]
    #     lq1_img = lq_img_3D[i-2:i+4]
    #     #lq2_img = lq2_img_3D[i-2:i+4]
        
    #     img_gt = center_crop_arr(img_gt, 512)
    #     lq1_img = center_crop_arr(lq1_img, 512)
    #     #lq2_img = center_crop_arr(lq2_img, 512)

    #     gt = img_gt.astype(np.float32)#(img_gt * 2 - 1).astype(np.float32)

    #     #[0, 1]
    #     x = lq1_img.astype(np.float32)

    #     target_all.append(gt)
    #     source_all.append(x)
    
    # target_all = np.array(target_all)
    # source_all = np.array(source_all)
        # length = pil_img_3D.shape[0]
        # median = round(length / 2)
        # for index_A in range(median-20, median+20):
        
        # #index_A = random.randint(median-40, median+50)
        # img_gt = pil_img_3D[index-2: index+4]
        # lq_single = transformed_img_3D[index-2: index+4]
        # img_gt = target_all[index][2]
        # lq_single= source_all[index][2]
        # lq_ori = np.stack((lq_single, lq_single, lq_single), axis=0)
        # gt_ori = np.stack((img_gt, img_gt, img_gt), axis=0)

  

        # img_gt = pil_img_3D[index_A-2:index_A+4]
        # lq_img = lq_img_3D[index_A-2:index_A+4]

    # target_all = []
    # source_all = []
    # #for i in range(median-40, median+50):
    # for i in range(10, 195):
    #     img_gt = pil_img_3D[i-2:i+4]
    #     lq1_img = lq_img_3D[i-2:i+4]
    #     #lq2_img = lq2_img_3D[i-2:i+4]
        

        # max_value, min_value = img_gt.max(), img_gt.min() 
        # gt_bef = (img_gt - min_value) / (max_value - min_value)

        # max_value, min_value = lq_single.max(), lq_single.min() 
        # lq_bef = (lq_single - min_value) / (max_value - min_value)

        # lq_bef = np.stack((lq_bef, lq_bef, lq_bef), axis=0)
        # gt_bef = np.stack((gt_bef, gt_bef, gt_bef), axis=0)

        # lq_2 = np.array([lq_bef])
        # gt_2 = np.array([gt_bef])
        # psnr_lq_bef = calculate_psnr_pt(lq_2, gt_2, crop_border=0).mean()
        # psnr_all_lq_bef.append(np.float32(psnr_lq_bef))

        # img_gt = center_crop_arr(img_gt, 512)
        # lq_single = center_crop_arr(lq_single, 512)
    #     #lq2_img = center_crop_arr(lq2_img, 512)

    #     gt = img_gt.astype(np.float32)#(img_gt * 2 - 1).astype(np.float32)

    #     #[0, 1]
    #     x = lq1_img.astype(np.float32)

    #     target_all.append(gt)
    #     source_all.append(x)
    
    # target_all = np.array(target_all)
    # source_all = np.array(source_all)

    # pbar = tqdm(range(target_all.shape[0]))
    # #for file_path in pbar:

    # for index in pbar:
    #     x = source_all[index]  #(6,512,512)
    #     gt= target_all[index]


        #pbar.set_description(file_path)
        # save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        # parent_path, stem, _ = get_file_name_parts(save_path)
        # save_path = os.path.join(parent_path, f"{stem}.png")
        # if os.path.exists(save_path):
        #     if args.skip_if_exist:
        #         print(f"skip {save_path}")
        #         continue
        #     else:
        #         raise RuntimeError(f"{save_path} already exist")
        # os.makedirs(parent_path, exist_ok=True)
        
        # load low-quality image and resize
        # lq = Image.open(file_path).convert("RGB")
        # gt = Image.open(file_path.replace("lq","hq")).convert("RGB")
        # gt = np.array(gt)
        # gt = gt/255#.clamp(0,1)
        
        
        #gt = einops.rearrange(gt, "h w c -> c h w")

        # if args.sr_scale != 1:
        #     lq = lq.resize(
        #         tuple(int(x * args.sr_scale) for x in lq.size), Image.BICUBIC
        #     )
        #lq_resized = auto_resize(lq, args.image_size)
        # lq_resized = center_crop_arr(x, args.image_size) #(512,512)
        # gt = center_crop_arr(gt, args.image_size) #(512,512)
            
        # max_value, min_value = x[2].max(), x[2].min() 
        # x_t = (x[2] - min_value) / (max_value - min_value)

        # max_value, min_value = gt[2].max(), gt[2].min() 
        # gt_t = (gt[2] - min_value) / (max_value - min_value)

        # gt_list.append(gt[2])
        # lq_list.append(x[2])
        
        #lq = np.stack((x[2], x[2], x[2]), axis=0)  #(3,512,512)
        # x_t = np.stack((x_t, x_t, x_t), axis=2)
        
        # x_t = (x_t * 255).clip(0, 255).astype(np.uint8)
        # Image.fromarray(x_t).save(path_lq)
        #lq_resized = center_crop_arr_new(x, args.image_size) #(512,512)
        img_gt = center_crop_arr(img_gt, 512)
        lq_single = center_crop_arr(lq_single, 512)


        # lq = lq_img_3D.transpose(2,0,1)
        # lq_resized = lq_img_3D
        # gt = pil_img_3D.transpose(2,0,1)

        #lq_resized = np.stack((x, x, x), axis=-1)  ##(6,512,512,3)

        #lq = x
        #gt = np.stack((gt[2], gt[2], gt[2]), axis=0)  #(3,512,512)

        # """Normalize img to [0,1]
        # """
        max_value, min_value = lq_single.max(), lq_single.min() 
        lq_single = (lq_single - min_value) / (max_value - min_value)

        max_value, min_value = img_gt.max(), img_gt.min() 
        img_gt = (img_gt - min_value) / (max_value - min_value)

        # max_value, min_value = lq.max(), lq.min() 
        # lq = (lq - min_value) / (max_value - min_value)
    
        lq = np.stack((lq_single, lq_single, lq_single), axis=0)
        lq_resized = np.stack((lq_single, lq_single, lq_single), axis=-1)
        
        gt = np.stack((img_gt, img_gt, img_gt), axis=0)  #[3,6,512,512]

        # padding
        #x = pad(np.array(lq_resized), scale=64)
        x = lq_resized
        x = torch.tensor(x, dtype=torch.float32, device=device)#.unsqueeze(0).contiguous() 
        #x = x.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  #[1, 3, 512, 512]
        x = x.permute(2, 0, 1).unsqueeze(0).contiguous()  #[1, 3, 512, 512]
    #x = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
# x = torch.tensor(x, dtype=torch.float32, device=device)
# x = x.permute(0, 3, 1, 2).unsqueeze(0)#.contiguous()  #[1,6, 3, 512, 512]
# x = torch.cat(tuple(12*[x]), axis=0).contiguous()
        try:
            # pred = model(x).detach().squeeze(0).permute(1, 2, 0) * 255
            # pred = pred.clamp(0, 255).to(torch.uint8).cpu().numpy()
            #pred = model(x).detach().squeeze(0).permute(1, 2, 0)
            
            #pred, heatmap, heatmap_ori = model(x)  #[12, 180, 64, 64]
            pred = model(x)
            #pred, decision, quality_score, vis_mask = model(x,train_mode=False)  #vis_mask=[48,64]
            #pred, decision, quality_score, img_routing_score = model(x,train_mode=False)
            #pred= model(x,train_mode=False)
            
            pred = pred.detach().squeeze(0)  #[12,3,512,512]
            # heatmap = heatmap.detach().squeeze(0)  #[12,3,512,512]
            # heatmap_ori = heatmap_ori.detach().squeeze(0)
            
            pred = pred.clamp(0, 1).cpu().numpy()
            #heatmap = heatmap.clamp(0, 1).cpu().numpy()
            # heatmap = heatmap.cpu().numpy()
            # heatmap_ori = heatmap_ori.cpu().numpy()
            import torch.nn.functional as F
            # a = F.mse_loss(input=torch.tensor(lq), target=torch.tensor(gt))#.sum(dim=(1,2,3))
            # b = F.mse_loss(input=torch.tensor(pred), target=torch.tensor(gt))#.sum(dim=(1,2,3))
            # a = torch.abs(torch.tensor(lq)-torch.tensor(gt)).sum(dim=(0,1,2))
            # b = torch.abs(torch.tensor(pred)-torch.tensor(gt)).sum(dim=(0,1,2))

        except RuntimeError as e:
            print(f"inference failed, error: {e}")
        # continue
        
    # pred_list.append(pred[0][0])
    # gt_list.append(gt[0])
    # lq_list.append(lq[0])

    #pred = center_expand_arr(pred, [3,311,260])  #expand后有问题
    #pred = pred[0].transpose(1,2,0)  #[311,260,3]
        pred = pred.transpose(1,2,0)
    #heatmap = heatmap[0].transpose(1,2,0)
        # gt = gt[:,2,...]
        # lq = lq[:,2,...]
        gt_img = gt.transpose(1,2,0)  #[311,260,3]
        lq_img = lq.transpose(1,2,0)  #[311,260,3]



    # pixel_diff = cv2.absdiff(gt_img,lq_img)
    # pixel_diff = np.array(pixel_diff)[:,:,0]
    # _range = np.max(pixel_diff) - np.min(pixel_diff) 
    # pixel_diff = (pixel_diff - np.min(pixel_diff)) / _range

    # heatmap = np.array(heatmap)
    # _range = np.max(heatmap) - np.min(heatmap) 
    # heatmap = (heatmap - np.min(heatmap)) / _range

    #heatmap_ori = np.array(heatmap_ori)
    # for i in range(heatmap_ori.shape[0]):
    #     _range = np.max(heatmap_ori[i]) - np.min(heatmap_ori[i]) 
    #     heatmap_single = (heatmap_ori[i] - np.min(heatmap_ori[i])) / _range
    #     heatmap_single = (heatmap_single * 255).clip(0, 255).astype(np.uint8)
    #     path_heatmap = os.path.join(args.output, f'ori_heatmap_{index}_{i}.png')
    
    #     plt.subplot(1, 1, 1)
    #     plt.imshow(heatmap_single,cmap='jet')
    #     plt.axis('off')
    #     plt.savefig(path_heatmap, format='png', dpi=600,bbox_inches='tight',pad_inches = -0.01)
    
        pred_img = (pred * 255).clip(0, 255).astype(np.uint8)
    #heatmap = (heatmap * 255).clip(0, 255).astype(np.uint8)
        gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)
        lq_img = (lq_img * 255).clip(0, 255).astype(np.uint8)

    # heatmap = np.array(heatmap)[:,:,0]

    # path_heatmap = os.path.join(args.output, f'heatmap_{index}_8.png')
    
    # plt.subplot(1, 1, 1)
    # plt.imshow(heatmap,cmap='jet')
    # plt.axis('off')
    # plt.clim(0.0, 0.7)
    # plt.savefig(path_heatmap, format='png', dpi=600,bbox_inches='tight',pad_inches = -0.01)
    
    
    # path_diff = os.path.join(args.output, f'diff_{index}.png')
    # plt.subplot(1, 1, 1)
    # plt.imshow(pixel_diff,cmap='jet')
    # plt.axis('off')
    # plt.clim(0.05, 0.3)
    # plt.savefig(path_diff, format='png', dpi=600,bbox_inches='tight',pad_inches = -0.01)
    
    #pred = pred.detach().cpu().numpy()
        pred = einops.rearrange(pred, "h w c -> c h w")
        lpips_metric = LPIPS(net="alex")
    #pred = einops.rearrange(pred, "h w c -> c h w")
        pred_2 = np.array([pred])  #[1,3,512,512]
        pred_ = torch.tensor(pred , dtype=torch.float32)

        lq_2 = np.array([lq])
        lq_ = torch.tensor(lq , dtype=torch.float32)
        
        gt_2 = np.array([gt])
        gt_ = torch.tensor(gt, dtype=torch.float32)

    # lq_2_ori = np.array([lq_ori])
    # lq_ori_ = torch.tensor(lq_ori , dtype=torch.float32)
    
    # gt_2_ori = np.array([gt_ori])
    # gt_ori_ = torch.tensor(gt_ori, dtype=torch.float32)
    
        psnr = calculate_psnr_pt(pred_2, gt_2, crop_border=0).mean()
        lpips = lpips_metric(pred_, gt_, normalize=True).mean()  #0.2679
        ssim = calculate_ssim_pt(pred_2, gt_2, crop_border=0).mean()  #0.7713
    # FID = fid.calculate_fid(gt_2.transpose(0,2,3,1), pred_2.transpose(0,2,3,1), False, 1)
    # musiq_metric = pyiqa.create_metric('musiq',device='cuda')
    # MUSIQ = musiq_metric(pred_.unsqueeze(0)).detach().cpu().numpy()[0][0]
    # niqe_metric = pyiqa.create_metric('niqe',device='cuda')
    # NIQE = niqe_metric(pred_.unsqueeze(0)).detach().cpu().numpy()
    # NRMSE = calculate_nrmse(pred, gt)

        psnr_all.append(np.float32(psnr))
        lpips_all.append(np.float32(lpips))
        ssim_all.append(np.float32(ssim))
    # fid_all.append(np.float32(FID))
    # niqe_all.append(np.float32(NIQE))
    # musiq_all.append(np.float32(MUSIQ))
    # nrmse_all.append(np.float32(NRMSE))

        psnr_lq = calculate_psnr_pt(lq_2, gt_2, crop_border=0).mean()
        lpips_lq = lpips_metric(lq_, gt_, normalize=True).mean()  #0.2679
        ssim_lq = calculate_ssim_pt(lq_2, gt_2, crop_border=0).mean()  #0.7713

    # psnr_lq = calculate_psnr_pt(lq_2_ori, gt_2_ori, crop_border=0).mean()
    # lpips_lq = lpips_metric(lq_ori_, gt_ori_, normalize=True).mean()  #0.2679
    # ssim_lq = calculate_ssim_pt(lq_2_ori, gt_2_ori, crop_border=0).mean()  #0.7713

    # FID_lq = fid.calculate_fid(gt_2.transpose(0,2,3,1), lq_2.transpose(0,2,3,1), False, 1)
    # musiq_metric = pyiqa.create_metric('musiq',device='cuda')
    # MUSIQ_lq = musiq_metric(lq_.unsqueeze(0)).detach().cpu().numpy()[0][0]
    # niqe_metric = pyiqa.create_metric('niqe',device='cuda')
    # NIQE_lq = niqe_metric(lq_.unsqueeze(0)).detach().cpu().numpy()
    # NRMSE_lq = calculate_nrmse(pred, lq)

        psnr_all_lq.append(np.float32(psnr_lq))
        lpips_all_lq.append(np.float32(lpips_lq))
        ssim_all_lq.append(np.float32(ssim_lq))
    # fid_all_lq.append(np.float32(FID_lq))
    # niqe_all_lq.append(np.float32(NIQE_lq))
    # musiq_all_lq.append(np.float32(MUSIQ_lq))
    # nrmse_all_lq.append(np.float32(NRMSE_lq))

        path = os.path.join(args.output, f'pred_{index+1}_psnr({psnr}_ssim({ssim})_lpips({lpips}.png')
        path_gt = os.path.join(args.output, f'gt_{index+1}.png')
        
        path_lq = os.path.join(args.output, f'lq_{index+1}_psnr({psnr_lq}_ssim({ssim_lq})_lpips({lpips_lq}.png')
        Image.fromarray(pred_img).save(path)

    #cv2.imwrite(path_heatmap, heatmap)
    #Image.fromarray(heatmap,mode='RGB').save(path_heatmap)
        Image.fromarray(gt_img).save(path_gt)
        Image.fromarray(lq_img).save(path_lq)

    
    # pred_arr_3d = np.array(pred_list)
    # gt_arr_3d = np.array(gt_list)
    # lq_arr_3d = np.array(lq_list)

    # pred_img_3d = sitk.GetImageFromArray(pred_arr_3d)
    # sitk.WriteImage(pred_img_3d, os.path.join(args.output, 'pred_s1_3d_103.nii.gz'))
    # gt_img_3d = sitk.GetImageFromArray(gt_arr_3d)
    # sitk.WriteImage(gt_img_3d, os.path.join(args.output, 'gt_s1_3d_103.nii.gz'))
    # lq_img_3d = sitk.GetImageFromArray(lq_arr_3d)
    # sitk.WriteImage(lq_img_3d, os.path.join(args.output, 'lq_s1_3d_103.nii.gz'))

    print("psnr:",np.mean(psnr_all),np.std(psnr_all))
    print("ssim:",np.mean(ssim_all),np.std(ssim_all))
    print("lpips:",np.mean(lpips_all),np.std(lpips_all))
    # print("fid:",np.mean(fid_all),np.std(fid_all))
    # print("musiq:",np.mean(musiq_all),np.std(musiq_all))
    # print("niqe:",np.mean(niqe_all),np.std(niqe_all))

    print("psnr_lq:",np.mean(psnr_all_lq),np.std(psnr_all_lq))
    # print("psnr_lq_bef:",np.mean(psnr_all_lq_bef),np.std(psnr_all_lq_bef))
    print("ssim_lq:",np.mean(ssim_all_lq),np.std(ssim_all_lq))
    print("lpips_lq:",np.mean(lpips_all_lq),np.std(lpips_all_lq))
    # print("fid_lq:",np.mean(fid_all_lq),np.std(fid_all_lq))
    # print("musiq_lq:",np.mean(musiq_all_lq),np.std(musiq_all_lq))
    # print("niqe_lq:",np.mean(niqe_all_lq),np.std(niqe_all_lq))

# @torch.no_grad()
# def main():
#     args = parse_args()
#     #pl.seed_everything(args.seed)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model: pl.LightningModule = instantiate_from_config(OmegaConf.load(args.config))
#     a = torch.load(args.ckpt, map_location="cpu")
#     load_state_dict(model, a, strict=True)
#     model.freeze()
#     #frozen_module(model)
#     model.to(device)
#     psnr_all = []
#     ssim_all = []
#     lpips_all = []
#     fid_all = []
#     niqe_all = []
#     musiq_all = []
#     nrmse_all = []

#     psnr_all_lq = []
#     ssim_all_lq = []
#     lpips_all_lq = []
#     fid_all_lq = []
#     niqe_all_lq = []
#     musiq_all_lq = []
#     nrmse_all_lq = []

#     pred_list = []
#     gt_list = []
#     lq_list = []
#     idx = 0
#     target_all = []
#     source_all = []
#     #assert os.path.isdir(args.input)

#     #target_all, source_all = get_data(args.file_list) #list, [512,512,3]*n*80

#     paths = load_file_list('/public_bme/data/lifeng/data/val_hcp.list')
#     for index in range(len(paths)):
#         gt_path = paths[index]
#         try:
#             pil_img_3D = load_nii(gt_path)
            
#         except:
#             continue
#         # length = pil_img_3D.shape[0]
#         # median = round(length / 2)

#         # index_A = random.randint(median-40, median+50)
#         # pil_img_3D = pil_img_3D[index_A]
#         # pil_img_3D = np.stack((pil_img_3D, pil_img_3D, pil_img_3D), axis=0)
#         # pil_img_3D = center_crop_arr(pil_img_3D, 256, dim=3)
#         transformed_img_3D = sim_motion(pil_img_3D)
#         # transformed_img_3D_ = normalize_mri_volume(transformed_img_3D)
#         # max_value, min_value = transformed_img_3D_.max(), transformed_img_3D_.min() 
#         # transformed_img_3D_ = (transformed_img_3D_ - min_value) / (max_value - min_value)

#         # pil_img_3D_ = normalize_mri_volume(pil_img_3D)
#         # max_value, min_value = pil_img_3D_.max(), pil_img_3D_.min() 
#         # pil_img_3D_ = (pil_img_3D_ - min_value) / (max_value - min_value)

#         # img_gt_bef = pil_img_3D_[2]
#         # lq_single_bef = transformed_img_3D_[2]

#         # lq_single_bef = np.stack((lq_single_bef, lq_single_bef, lq_single_bef), axis=0)
#         # img_gt_bef = np.stack((img_gt_bef, img_gt_bef, img_gt_bef), axis=0)

#         # max_value, min_value = transformed_img_3D.max(), transformed_img_3D.min() 
#         # lq_single = (transformed_img_3D - min_value) / (max_value - min_value)

#         # max_value, min_value = pil_img_3D.max(), pil_img_3D.min() 
#         # img_gt = (pil_img_3D - min_value) / (max_value - min_value)

#         # lq_single_2 = np.array([lq_single])
#         # img_gt_2 = np.array([img_gt])

#         # psnr_lq_bef_2 = calculate_psnr_pt(lq_single_2, img_gt_2, crop_border=0).mean()
#         # c = torch.abs(torch.tensor(lq_single_2)-torch.tensor(img_gt_2)).mean(dim=(0,1,2,3))


#         # from utils import util_image
   
#         # pil_img_3D = util_image.imread('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/gt_0_150.png',
#         #                         chn = 'rgb', dtype='float32')
#         # lq_img_3D = util_image.imread('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/lq_0_150_psnr(79.9731978784669_ssim(0.9999999704552437)_lpips(1.4173985540821832e-08.png',
#         #                         chn = 'rgb', dtype='float32')

#         # max_value, min_value = lq_img_3D.max(), lq_img_3D.min() 
#         # lq_img_3D = (lq_img_3D - min_value) / (max_value - min_value)

#         # max_value, min_value = pil_img_3D.max(), pil_img_3D.min() 
#         # pil_img_3D = (pil_img_3D - min_value) / (max_value - min_value)

#     # pil_img_3D_selected = sitk.GetImageFromArray(pil_img_3D[10:195])
#     # sitk.WriteImage(pil_img_3D_selected, os.path.join(args.output, 'gt_s1_3d_352738_ori.nii.gz'))

#     # lq_img_3D_selected = sitk.GetImageFromArray(lq_img_3D[10:195])
#     # sitk.WriteImage(lq_img_3D_selected, os.path.join(args.output, 'lq_s1_3d_352738_ori.nii.gz'))
    
#     #pbar = tqdm(list_image_files_natural(args.input, follow_links=True))

#     #----- Test -----#
#     # pil_img_3D = load_nii('/public_bme/data/lifeng/data/moco/openneuro/sub-000175/anat/sub-000175_acq-standard_T1w.nii.gz')
#     # lq_img_3D = load_nii('/public_bme/data/lifeng/data/moco/openneuro/sub-000175/anat/sub-000175_acq-headmotion1_T1w.nii.gz')
#     ##index_A = random.randint(130, 215) #[179, 180] #random.sample(range(168, 188), 2)
#     # target_all = []
#     # source_all = []
#     # for i in range(138,195):
#     #     img_gt = pil_img_3D[i-2:i+4]
#     #     lq1_img = lq_img_3D[i-2:i+4]
#     #     #lq2_img = lq2_img_3D[i-2:i+4]
#         # img_gt_af = pil_img_3D[1]
#         # lq_single_af = transformed_img_3D[1]
#         # img_gt_af = center_crop_arr(np.array(img_gt_af), 512, dim=2)
#         # lq_single_af = center_crop_arr(np.array(lq_single_af), 512, dim=2)
#     #     lq1_img = center_crop_arr(lq1_img, 512)
#     #     #lq2_img = center_crop_arr(lq2_img, 512)

#     #     gt = img_gt.astype(np.float32)#(img_gt * 2 - 1).astype(np.float32)

#     #     #[0, 1]
#     #     x = lq1_img.astype(np.float32)

#     #     target_all.append(gt)
#     #     source_all.append(x)
    
#     # target_all = np.array(target_all)
#     # source_all = np.array(source_all)
        

#         length = pil_img_3D.shape[0]
#         median = round(length / 2)
#         # # for index_A in range(median-20, median+20):
#         for _ in range(2):
#             index_A = random.randint(median-40, median+50)
#             img_gt = pil_img_3D[index_A-2:index_A+4]
#             lq_single = transformed_img_3D[index_A-2:index_A+4]

#             # max_value, min_value = lq_single.max(), lq_single.min() 
#             # lq_single_bef = (lq_single - min_value) / (max_value - min_value)

#             # max_value, min_value = img_gt.max(), img_gt.min() 
#             # img_gt_bef = (img_gt - min_value) / (max_value - min_value)

#         #     img_gt_bef = pil_img_3D_[index_A]
#         #     lq_single_bef = transformed_img_3D_[index_A]

#         #     lq_single_bef = np.stack((lq_single_bef, lq_single_bef, lq_single_bef), axis=0)
#         #     img_gt_bef = np.stack((img_gt_bef, img_gt_bef, img_gt_bef), axis=0)

#         #     lq_single_bef_2 = np.array([lq_single_bef])
#         #     img_gt_bef_2 = np.array([img_gt_bef])

#         #     psnr_lq_bef_2 = calculate_psnr_pt(lq_single_bef_2, img_gt_bef_2, crop_border=0).mean()
#         #     c = torch.abs(torch.tensor(lq_single_bef_2)-torch.tensor(img_gt_bef_2)).mean(dim=(0,1,2,3))
#         # lq_ori = np.stack((lq_single, lq_single, lq_single), axis=0)
#         # gt_ori = np.stack((img_gt, img_gt, img_gt), axis=0)

#         # max_value, min_value = gt_ori.max(), gt_ori.min() 
#         # gt_ori = (gt_ori - min_value) / (max_value - min_value)

#         # max_value, min_value = lq_ori.max(), lq_ori.min() 
#         # lq_ori = (lq_ori - min_value) / (max_value - min_value)

#         # img_gt = pil_img_3D[index_A-2:index_A+4]
#         # lq_img = lq_img_3D[index_A-2:index_A+4]

#     # target_all = []
#     # source_all = []
#     # #for i in range(median-40, median+50):
#     # for i in range(10, 195):
#     #     img_gt = pil_img_3D[i-2:i+4]
#     #     lq1_img = lq_img_3D[i-2:i+4]
#     #     #lq2_img = lq2_img_3D[i-2:i+4]
        
#             img_gt = center_crop_arr(img_gt, 512, dim=3)
#             lq_single = center_crop_arr(lq_single, 512, dim=3)

#             target_all.append(img_gt)
#             source_all.append(lq_single)

#     #     #lq2_img = center_crop_arr(lq2_img, 512)

#     #     gt = img_gt.astype(np.float32)#(img_gt * 2 - 1).astype(np.float32)

#     #     #[0, 1]
#     #     x = lq1_img.astype(np.float32)

#     #     target_all.append(gt)
#     #     source_all.append(x)
    
#     # target_all = np.array(target_all)
#     # source_all = np.array(source_all)

#     # pbar = tqdm(range(target_all.shape[0]))
#     # #for file_path in pbar:

#     # for index in pbar:
#     #     x = source_all[index]  #(6,512,512)
#     #     gt= target_all[index]


#         #pbar.set_description(file_path)
#         # save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
#         # parent_path, stem, _ = get_file_name_parts(save_path)
#         # save_path = os.path.join(parent_path, f"{stem}.png")
#         # if os.path.exists(save_path):
#         #     if args.skip_if_exist:
#         #         print(f"skip {save_path}")
#         #         continue
#         #     else:
#         #         raise RuntimeError(f"{save_path} already exist")
#         # os.makedirs(parent_path, exist_ok=True)
        
#         # load low-quality image and resize
#         # lq = Image.open(file_path).convert("RGB")
#         # gt = Image.open(file_path.replace("lq","hq")).convert("RGB")
#         # gt = np.array(gt)
#         # gt = gt/255#.clamp(0,1)
        
        
#         #gt = einops.rearrange(gt, "h w c -> c h w")

#         # if args.sr_scale != 1:
#         #     lq = lq.resize(
#         #         tuple(int(x * args.sr_scale) for x in lq.size), Image.BICUBIC
#         #     )
#         #lq_resized = auto_resize(lq, args.image_size)
#         # lq_resized = center_crop_arr(x, args.image_size) #(512,512)
#         # gt = center_crop_arr(gt, args.image_size) #(512,512)
            
#         # max_value, min_value = x[2].max(), x[2].min() 
#         # x_t = (x[2] - min_value) / (max_value - min_value)

#         # max_value, min_value = gt[2].max(), gt[2].min() 
#         # gt_t = (gt[2] - min_value) / (max_value - min_value)

#         # gt_list.append(gt[2])
#         # lq_list.append(x[2])
        
#         #lq = np.stack((x[2], x[2], x[2]), axis=0)  #(3,512,512)
#         # x_t = np.stack((x_t, x_t, x_t), axis=2)
        
#         # x_t = (x_t * 255).clip(0, 255).astype(np.uint8)
#         # Image.fromarray(x_t).save(path_lq)
#         #lq_resized = center_crop_arr_new(x, args.image_size) #(512,512)
#     # img_gt = center_crop_arr(img_gt, args.image_size)
#     # lq_single = center_crop_arr(lq_img, args.image_size)
        
#             img_gt = img_gt[2]
#             lq_single = lq_single[2]
#             lq = np.stack((lq_single, lq_single, lq_single), axis=0)
#             lq_resized = np.stack((lq_single, lq_single, lq_single), axis=-1)
#             gt = np.stack((img_gt, img_gt, img_gt), axis=0)

#             max_value, min_value = gt.max(), gt.min() 
#             gt = (gt - min_value) / (max_value - min_value)

#             max_value, min_value = lq.max(), lq.min() 
#             lq = (lq - min_value) / (max_value - min_value)

#             max_value, min_value = lq_resized.max(), lq_resized.min() 
#             lq_resized = (lq_resized - min_value) / (max_value - min_value)        
    
#                 # gt = np.stack((img_gt, img_gt, img_gt), axis=0)
#             # gt = np.array(img_gt)
#             # lq = lq_single
#             # lq_resized = lq.transpose(1,2,0)  #(512,512,3)

#             x = lq_resized
#             #x = lq_single_af
#             x = torch.tensor(x, dtype=torch.float32, device=device)#.unsqueeze(0).contiguous() 
#             x = x.permute(2, 0, 1).unsqueeze(0).contiguous()  #[1, 3, 512, 512]
#             #x = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
#         # x = torch.tensor(x, dtype=torch.float32, device=device)
#         # x = x.permute(0, 3, 1, 2).unsqueeze(0)#.contiguous()  #[1,6, 3, 512, 512]
#         # x = torch.cat(tuple(12*[x]), axis=0).contiguous()
#             try:
#                 # pred = model(x).detach().squeeze(0).permute(1, 2, 0) * 255
#                 # pred = pred.clamp(0, 255).to(torch.uint8).cpu().numpy()
#                 #pred = model(x).detach().squeeze(0).permute(1, 2, 0)
                
#                 #pred, heatmap, heatmap_ori = model(x)  #[12, 180, 64, 64]
#                 #pred = model(x)
#                 pred, decision, quality_score, vis_mask = model(x,train_mode=False)  #vis_mask=[48,64]
#                 #pred, decision, quality_score, img_routing_score = model(x,train_mode=False)
#                 #pred= model(x,train_mode=False)
                
#                 pred = pred.detach().squeeze(0)  #[12,3,512,512]
#                 # heatmap = heatmap.detach().squeeze(0)  #[12,3,512,512]
#                 # heatmap_ori = heatmap_ori.detach().squeeze(0)
                
#                 pred = pred.clamp(0, 1).cpu().numpy()
#                 #heatmap = heatmap.clamp(0, 1).cpu().numpy()
#                 # heatmap = heatmap.cpu().numpy()
#                 # heatmap_ori = heatmap_ori.cpu().numpy()
#                 import torch.nn.functional as F
#                 # a = F.mse_loss(input=torch.tensor(lq), target=torch.tensor(gt))#.sum(dim=(1,2,3))
#                 # b = F.mse_loss(input=torch.tensor(pred), target=torch.tensor(gt))#.sum(dim=(1,2,3))


#             except RuntimeError as e:
#                 print(f"inference failed, error: {e}")
#             # continue
            
#         # pred_list.append(pred[0][0])
#         # gt_list.append(gt[0])
#         # lq_list.append(lq[0])

#         #pred = center_expand_arr(pred, [3,311,260])  #expand后有问题
#         #pred = pred[0].transpose(1,2,0)  #[311,260,3]
#             pred = pred.transpose(1,2,0)
#         #heatmap = heatmap[0].transpose(1,2,0)
#             gt_img = gt.transpose(1,2,0)  #[311,260,3]
#             #lq_img = lq.transpose(1,2,0)  #[311,260,3]
#             lq_img = lq_resized


#         # pixel_diff = cv2.absdiff(gt_img,lq_img)
#         # pixel_diff = np.array(pixel_diff)[:,:,0]
#         # _range = np.max(pixel_diff) - np.min(pixel_diff) 
#         # pixel_diff = (pixel_diff - np.min(pixel_diff)) / _range

#         # heatmap = np.array(heatmap)
#         # _range = np.max(heatmap) - np.min(heatmap) 
#         # heatmap = (heatmap - np.min(heatmap)) / _range

#         #heatmap_ori = np.array(heatmap_ori)
#         # for i in range(heatmap_ori.shape[0]):
#         #     _range = np.max(heatmap_ori[i]) - np.min(heatmap_ori[i]) 
#         #     heatmap_single = (heatmap_ori[i] - np.min(heatmap_ori[i])) / _range
#         #     heatmap_single = (heatmap_single * 255).clip(0, 255).astype(np.uint8)
#         #     path_heatmap = os.path.join(args.output, f'ori_heatmap_{index}_{i}.png')
        
#         #     plt.subplot(1, 1, 1)
#         #     plt.imshow(heatmap_single,cmap='jet')
#         #     plt.axis('off')
#         #     plt.savefig(path_heatmap, format='png', dpi=600,bbox_inches='tight',pad_inches = -0.01)
        
#             pred_img = (pred * 255).clip(0, 255).astype(np.uint8)
#         #heatmap = (heatmap * 255).clip(0, 255).astype(np.uint8)
#             gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)
#             lq_img = (lq_img * 255).clip(0, 255).astype(np.uint8)

#         # heatmap = np.array(heatmap)[:,:,0]

#         # path_heatmap = os.path.join(args.output, f'heatmap_{index}_8.png')
        
#         # plt.subplot(1, 1, 1)
#         # plt.imshow(heatmap,cmap='jet')
#         # plt.axis('off')
#         # plt.clim(0.0, 0.7)
#         # plt.savefig(path_heatmap, format='png', dpi=600,bbox_inches='tight',pad_inches = -0.01)
        
        
#         # path_diff = os.path.join(args.output, f'diff_{index}.png')
#         # plt.subplot(1, 1, 1)
#         # plt.imshow(pixel_diff,cmap='jet')
#         # plt.axis('off')
#         # plt.clim(0.05, 0.3)
#         # plt.savefig(path_diff, format='png', dpi=600,bbox_inches='tight',pad_inches = -0.01)
        
#         #pred = pred.detach().cpu().numpy()
#             pred = einops.rearrange(pred, "h w c -> c h w")
#             lpips_metric = LPIPS(net="alex")
#         #pred = einops.rearrange(pred, "h w c -> c h w")
#             pred_2 = np.array([pred])  #[1,3,512,512]
#             pred_ = torch.tensor(pred , dtype=torch.float32)

#             lq_2 = np.array([lq])
#             lq_ = torch.tensor(lq , dtype=torch.float32)
            
#             gt_2 = np.array([gt])
#             gt_ = torch.tensor(gt, dtype=torch.float32)

#         # lq_2_ori = np.array([lq_ori])
#         # lq_ori_ = torch.tensor(lq_ori , dtype=torch.float32)
        
#         # gt_2_ori = np.array([gt_ori])
#         # gt_ori_ = torch.tensor(gt_ori, dtype=torch.float32)
        
#             psnr = calculate_psnr_pt(pred_2, gt_2, crop_border=0).mean()
#             lpips = lpips_metric(pred_, gt_, normalize=True).mean()  #0.2679
#             ssim = calculate_ssim_pt(pred_2, gt_2, crop_border=0).mean()  #0.7713
#         # FID = fid.calculate_fid(gt_2.transpose(0,2,3,1), pred_2.transpose(0,2,3,1), False, 1)
#         # musiq_metric = pyiqa.create_metric('musiq',device='cuda')
#         # MUSIQ = musiq_metric(pred_.unsqueeze(0)).detach().cpu().numpy()[0][0]
#         # niqe_metric = pyiqa.create_metric('niqe',device='cuda')
#         # NIQE = niqe_metric(pred_.unsqueeze(0)).detach().cpu().numpy()
#         # NRMSE = calculate_nrmse(pred, gt)

#             psnr_all.append(np.float32(psnr))
#             lpips_all.append(np.float32(lpips))
#             ssim_all.append(np.float32(ssim))
#         # fid_all.append(np.float32(FID))
#         # niqe_all.append(np.float32(NIQE))
#         # musiq_all.append(np.float32(MUSIQ))
#         # nrmse_all.append(np.float32(NRMSE))

#             psnr_lq = calculate_psnr_pt(lq_2, gt_2, crop_border=0).mean()
#             lpips_lq = lpips_metric(lq_, gt_, normalize=True).mean()  #0.2679
#             ssim_lq = calculate_ssim_pt(lq_2, gt_2, crop_border=0).mean()  #0.7713

#             # a = torch.abs(torch.tensor(lq)-torch.tensor(gt)).mean(dim=(0,1,2))
#             # b = torch.abs(torch.tensor(pred)-torch.tensor(gt)).mean(dim=(0,1,2))
#         # psnr_lq = calculate_psnr_pt(lq_2_ori, gt_2_ori, crop_border=0).mean()
#         # lpips_lq = lpips_metric(lq_ori_, gt_ori_, normalize=True).mean()  #0.2679
#         # ssim_lq = calculate_ssim_pt(lq_2_ori, gt_2_ori, crop_border=0).mean()  #0.7713

#         # FID_lq = fid.calculate_fid(gt_2.transpose(0,2,3,1), lq_2.transpose(0,2,3,1), False, 1)
#         # musiq_metric = pyiqa.create_metric('musiq',device='cuda')
#         # MUSIQ_lq = musiq_metric(lq_.unsqueeze(0)).detach().cpu().numpy()[0][0]
#         # niqe_metric = pyiqa.create_metric('niqe',device='cuda')
#         # NIQE_lq = niqe_metric(lq_.unsqueeze(0)).detach().cpu().numpy()
#         # NRMSE_lq = calculate_nrmse(pred, lq)

#             psnr_all_lq.append(np.float32(psnr_lq))
#             lpips_all_lq.append(np.float32(lpips_lq))
#             ssim_all_lq.append(np.float32(ssim_lq))
#         # fid_all_lq.append(np.float32(FID_lq))
#         # niqe_all_lq.append(np.float32(NIQE_lq))
#         # musiq_all_lq.append(np.float32(MUSIQ_lq))
#         # nrmse_all_lq.append(np.float32(NRMSE_lq))

#             path = os.path.join(args.output, f'pred_{idx}_psnr({psnr}_ssim({ssim})_lpips({lpips}.png') #_ssim({ssim})_lpips({lpips}
#             path_gt = os.path.join(args.output, f'gt_{idx}.png')
            
#             path_lq = os.path.join(args.output, f'lq_{idx}_psnr({psnr_lq}_ssim({ssim_lq})_lpips({lpips_lq}.png') #_ssim({ssim_lq})_lpips({lpips_lq}
#             Image.fromarray(pred_img).save(path)

#         #cv2.imwrite(path_heatmap, heatmap)
#         #Image.fromarray(heatmap,mode='RGB').save(path_heatmap)
#             Image.fromarray(gt_img).save(path_gt)
#             Image.fromarray(lq_img).save(path_lq)

#             idx += 1
    
#     # pred_arr_3d = np.array(pred_list)
#     # gt_arr_3d = np.array(gt_list)
#     # lq_arr_3d = np.array(lq_list)

#     # pred_img_3d = sitk.GetImageFromArray(pred_arr_3d)
#     # sitk.WriteImage(pred_img_3d, os.path.join(args.output, 'pred_s1_3d_103.nii.gz'))
#     # gt_img_3d = sitk.GetImageFromArray(gt_arr_3d)
#     # sitk.WriteImage(gt_img_3d, os.path.join(args.output, 'gt_s1_3d_103.nii.gz'))
#     # lq_img_3d = sitk.GetImageFromArray(lq_arr_3d)
#     # sitk.WriteImage(lq_img_3d, os.path.join(args.output, 'lq_s1_3d_103.nii.gz'))

#     print("psnr:",np.mean(psnr_all),np.std(psnr_all))
#     print("ssim:",np.mean(ssim_all),np.std(ssim_all))
#     print("lpips:",np.mean(lpips_all),np.std(lpips_all))
#     # print("fid:",np.mean(fid_all),np.std(fid_all))
#     # print("musiq:",np.mean(musiq_all),np.std(musiq_all))
#     # print("niqe:",np.mean(niqe_all),np.std(niqe_all))

#     print("psnr_lq:",np.mean(psnr_all_lq),np.std(psnr_all_lq))
#     print("ssim_lq:",np.mean(ssim_all_lq),np.std(ssim_all_lq))
#     print("lpips_lq:",np.mean(lpips_all_lq),np.std(lpips_all_lq))
#     # print("fid_lq:",np.mean(fid_all_lq),np.std(fid_all_lq))
#     # print("musiq_lq:",np.mean(musiq_all_lq),np.std(musiq_all_lq))
#     # print("niqe_lq:",np.mean(niqe_all_lq),np.std(niqe_all_lq))

#     np.save('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_1530_hq.npy', target_all)
#     np.save('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_1530_lq.npy', source_all)

if __name__ == "__main__":
    main()
    #print('The psnr is {}, lpips is {}, ssim is {}'.format(pnsr, lpips, ssim))
