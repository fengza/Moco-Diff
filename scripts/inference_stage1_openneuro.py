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
from utils.file import list_image_files, get_file_name_parts,list_image_files_natural
from utils.metrics import calculate_psnr_pt, LPIPS, calculate_ssim_pt
from utils.image import center_crop_arr
import fid
from inference_brain import get_data
import pyiqa
import statistics
import skimage.measure
import pandas as pd

def calculate_nrmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets)**2)) / (np.max(targets) - np.min(targets))

# def load_file_list(file_list_path: str):
#     files = []
#     # each line in file list contains a path of an image
#     with open(file_list_path, "r") as fin:
#         for line in fin:
#             path = line.strip()
#             if path:
#                 files.append(path)
#     return files

# def load_nii(nii_path):
#     try:
#         image = sitk.ReadImage(nii_path)
#     except:
#         assert False, f"failed to load image {nii_path}"
#     image_array = sitk.GetArrayFromImage(image)
    
    return image_array

# def preprocess_openneuro(files_gt_1, files_gt_2, files_lq_1, files_lq_2):
#     paths_gt_1 = load_file_list(files_gt_1)
#     paths_gt_2 = load_file_list(files_gt_2)
#     paths_lq_1 = load_file_list(files_lq_1)
#     paths_lq_2 = load_file_list(files_lq_2)

#     target_all_1 = []
#     target_all_2 = []
#     source_all_1 = []
#     source_all_2 = []

#     target_grades_1 = []
#     target_grades_2 = []
#     lq1_grades = []
#     lq2_grades = [] 

#     for index in range(len(paths_gt_1)): #num_samples
#         gt_path = paths_gt_1[index]
#         lq1_path = paths_lq_1[index]
#         #lq2_path = paths_lq_2[index]

#         pil_img_3D = load_nii(gt_path)
#         lq1_img_3D = load_nii(lq1_path)
#         #lq2_img_3D = load_nii(lq2_path)
        
#         names = gt_path.split('/')[:-2]
#         parent_path = '/'.join(names)
        
#         grade_file_path = parent_path + '/' + names[-1] + '_scans.tsv'

#         tsv_file = pd.read_csv(
#         grade_file_path,
#         sep='\t',
#         header=0,
#         index_col='filename'
#         )

#         tsv_file = tsv_file.values
#         target_grade = tsv_file[0][0]
#         lq1_grade = tsv_file[1][0]
#         #lq2_grade = tsv_file[2][0]

#         length = pil_img_3D.shape[0]
#         median = round(length / 2)
#         index_A = random.sample(range(168, 188), 2)

#         for i in index_A:

#             img_gt = pil_img_3D[i-2:i+4]
#             lq1_img = lq1_img_3D[i-2:i+4]
            
#             img_gt = center_crop_arr(img_gt, 512)
#             lq1_img = center_crop_arr(lq1_img, 512)
#             #lq2_img = center_crop_arr(lq2_img, 512)

#             target = img_gt.astype(np.float32)#(img_gt * 2 - 1).astype(np.float32)
#             target_all_1.append(target)
#             target_grades_1.append(target_grade)
            
#             #[0, 1]
#             source_1 = lq1_img.astype(np.float32)
#             source_all_1.append(source_1)
#             lq1_grades.append(lq1_grade)

#             # max_value, min_value = img_gt.max(), img_gt.min() 
#             # img_gt = (img_gt - min_value) / (max_value - min_value)

#             # max_value, min_value = lq1_img.max(), lq1_img.min() 
#             # lq1_img = (lq1_img - min_value) / (max_value - min_value)

#             # data = (img_gt[2] * 255.0).astype('uint8')  # 转换数据类型
#             # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
#             # # # 保存图片到本地
#             # imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/{index_A[0]}_hq.jpg', new_im_hq)

#             # data = (lq1_img[2] * 255.0).astype('uint8')  # 转换数据类型
#             # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
#             # # # 保存图片到本地
#             # imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/{index_A[0]}_lq.jpg', new_im_lq)


#             # source_2 = lq2_img.astype(np.float32)
#             # source_all_2.append(source_2)
#             # lq2_grades.append(lq2_grade)

#     for index in range(len(paths_gt_2)): #num_samples
#         gt_path = paths_gt_2[index]
#         lq2_path = paths_lq_2[index]
#         #lq2_path = paths_lq_2[index]

#         pil_img_3D = load_nii(gt_path)
#         lq2_img_3D = load_nii(lq2_path)
#         #lq2_img_3D = load_nii(lq2_path)

#         names = gt_path.split('/')[:-2]
#         parent_path = '/'.join(names)
        
#         grade_file_path = parent_path + '/' + names[-1] + '_scans.tsv'

#         tsv_file = pd.read_csv(
#         grade_file_path,
#         sep='\t',
#         header=0,
#         index_col='filename'
#         )

#         tsv_file = tsv_file.values
#         target_grade = tsv_file[0][0]
#         #lq1_grade = tsv_file[1][0]
#         try:
#             lq2_grade = tsv_file[2][0]
#         except:
#             lq2_grade = tsv_file[1][0]

#         length = pil_img_3D.shape[0]
#         median = round(length / 2)
#         index_A = random.sample(range(168, 188), 2)


#          # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
#         # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
#         # # # 保存图片到本地
#         # imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/{index}_{index_A}_hq.jpg', new_im_hq)

#         for i in index_A:
#             img_gt = pil_img_3D[i-2:i+4]
#             lq2_img = lq2_img_3D[i-2:i+4]
#             #lq2_img = lq2_img_3D[i-2:i+4]
            
#             img_gt = center_crop_arr(img_gt, 512)
#             lq2_img = center_crop_arr(lq2_img, 512)
#             #lq2_img = center_crop_arr(lq2_img, 512)

#             target = img_gt.astype(np.float32)#(img_gt * 2 - 1).astype(np.float32)
#             target_all_2.append(target)
#             target_grades_2.append(target_grade)
            
#             #[0, 1]
#             source_2 = lq2_img.astype(np.float32)
#             source_all_2.append(source_2)
#             lq2_grades.append(lq2_grade)
    
#     grades_dict_1 = {}
#     grades_dict_2 = {}
#     clean_all_1 = np.array(target_all_1)
#     clean_all_2 = np.array(target_all_2)
#     corrupt_all_1 = np.array(source_all_1)
#     corrupt_all_2 = np.array(source_all_2)
#     grades_dict_1['target_grades'] = target_grades_1
#     grades_dict_1['headmotion_grades'] = lq1_grades
#     grades_dict_2['target_grades'] = target_grades_2
#     grades_dict_2['headmotion_grades'] = lq2_grades
#     np.save('/public_bme/data/lifeng/data/moco/test/openneuro/data/hq_1.npy', clean_all_1)
#     np.save('/public_bme/data/lifeng/data/moco/test/openneuro/data/hq_2.npy', clean_all_2)
#     np.save('/public_bme/data/lifeng/data/moco/test/openneuro/data/lq_1.npy', corrupt_all_1)
#     np.save('/public_bme/data/lifeng/data/moco/test/openneuro/data/lq_2.npy', corrupt_all_2)
#     file_1 = pd.DataFrame(grades_dict_1)
#     file_2 = pd.DataFrame(grades_dict_2)
#     file_1.to_csv('/public_bme/data/lifeng/data/moco/test/openneuro/data/grades_1.csv')
#     file_2.to_csv('/public_bme/data/lifeng/data/moco/test/openneuro/data/grades_2.csv')
#     #return target_grades_1, target_grades_2, lq1_grades, lq2_grades


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,  default='/public_bme/data/lifeng/code/moco/TS_BHIR/configs/model/swinir.yaml', required=False)
    #parser.add_argument("--ckpt", type=str, default="/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints/lightning_logs/version_1285486/checkpoints/step=999.ckpt", required=False)#/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints
    parser.add_argument("--ckpt", type=str, default="/public_bme/data/lifeng/code/moco/TS_BHIR/checkpoints/lightning_logs/version_1400566/checkpoints/step=249.ckpt", required=False)
    #parser.add_argument("--ckpt", type=str, default="/public_bme/data/lifeng/code/moco/DiffBIR-main/checkpoints/lightning_logs/version_1386969/checkpoints/step=8349.ckpt", required=False)
    parser.add_argument("--file_list", type=str, default="/public_bme/data/lifeng/data/val_hcp.list", required=False)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_type", type=str, default="none")    
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--resize_back", action="store_true")
    parser.add_argument("--output", type=str,  default="/public_bme/data/lifeng/data/moco/test/openneuro/imgs", required=False)
    parser.add_argument("--skip_if_exist", action="store_true")
    parser.add_argument("--seed", type=int, default=231)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model: pl.LightningModule = instantiate_from_config(OmegaConf.load(args.config))
    a = torch.load(args.ckpt, map_location="cpu")
    load_state_dict(model, a, strict=True)
    model.freeze()
    model.to(device)
    psnr_all = []
    ssim_all = []
    lpips_all = []
    fid_all = []
    niqe_all = []
    musiq_all = []
    nrmse_all = []

    psnr_all_lq = []
    ssim_all_lq = []
    lpips_all_lq = []
    fid_all_lq = []
    niqe_all_lq = []
    musiq_all_lq = []
    nrmse_all_lq = []
    #assert os.path.isdir(args.input)
    gt_1_path = '/public_bme/data/lifeng/data/openneuro_gt_1.list'
    gt_2_path = '/public_bme/data/lifeng/data/openneuro_gt_2.list'
    lq_1_path = '/public_bme/data/lifeng/data/openneuro_lq_1.list'
    lq_2_path = '/public_bme/data/lifeng/data/openneuro_lq_2.list'


    # files_gt_1 = load_file_list(gt_1_path)
    # files_gt_2 = load_file_list(gt_2_path)
    # files_lq_1 = load_file_list(lq_1_path) 
    # files_lq_2 = load_file_list(lq_2_path)
    #preprocess_openneuro(gt_1_path, gt_2_path, lq_1_path, lq_2_path)
    #target_all, source_all = get_data(args.file_list,crop_type=args.crop_type) #list, [512,512,3]*n*80
    target_all_1 = np.load('/public_bme/data/lifeng/data/moco/test/openneuro/data/hq_1.npy')
    target_all_2 = np.load('/public_bme/data/lifeng/data/moco/test/openneuro/data/hq_2.npy')
    source_all_1 = np.load('/public_bme/data/lifeng/data/moco/test/openneuro/data/lq_1.npy')
    source_all_2 = np.load('/public_bme/data/lifeng/data/moco/test/openneuro/data/lq_2.npy')

    data_1 = pd.read_csv('/public_bme/data/lifeng/data/moco/test/openneuro/data/grades_1.csv')#['target_grades']
    data_2 = pd.read_csv('/public_bme/data/lifeng/data/moco/test/openneuro/data/grades_2.csv')
    target_grades_1 = list(data_1['target_grades'])
    target_grades_2 = list(data_1['headmotion_grades'])
    lq1_grades = list(data_2['target_grades']) 
    lq2_grades = list(data_2['headmotion_grades'])
    #lq_grades = lq1_grades + lq2_grades   #148 ✖️ 2 ✖️ 2
    #target_grades = target_grades + target_grades
    #pbar = tqdm(list_image_files_natural(args.input, follow_links=True))
    pbar = tqdm(range(target_all_1.shape[0]))
    #for file_path in pbar:
    # for num in range(2):
    #     source_all = f'source_all_{num+1}'
    for index in pbar:
        x = source_all_1[index]#[2]  #(512,512)
        gt= target_all_1[index]#[2]
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

        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(int(x * args.sr_scale) for x in lq.size), Image.BICUBIC
            )
        #lq_resized = auto_resize(lq, args.image_size)
        # lq_resized = center_crop_arr(x, args.image_size) #(512,512)
        # gt = center_crop_arr(gt, args.image_size) #(512,512)
        
        # lq_resized = x

        lq = np.stack((x[2], x[2], x[2]), axis=0)  #(3,512,512)
        #lq_resized = center_crop_arr_new(x, args.image_size) #(512,512)
        lq_resized = np.stack((x, x, x), axis=-1)  ##(6,512,512,3)
        # gt = center_crop_arr(gt, args.image_size)
        # lq = center_crop_arr(x, args.image_size)
        #lq = x
        gt = np.stack((gt[2], gt[2], gt[2]), axis=0)  #(3,512,512)
        
        """Normalize img to [0,1]
        """
        max_value, min_value = lq_resized.max(), lq_resized.min() 
        lq_resized = (lq_resized - min_value) / (max_value - min_value)

        max_value, min_value = gt.max(), gt.min() 
        gt = (gt - min_value) / (max_value - min_value)

        max_value, min_value = lq.max(), lq.min() 
        lq = (lq - min_value) / (max_value - min_value)
        # padding
        #x = pad(np.array(lq_resized), scale=64)
        x = lq_resized

        #x = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
        x = torch.tensor(x, dtype=torch.float32, device=device)
        x = x.permute(0, 3, 1, 2).unsqueeze(0)#.contiguous()  #[1,6, 3, 512, 512]
        x = torch.cat(tuple(12*[x]), axis=0).contiguous()
        try:
            # pred = model(x).detach().squeeze(0).permute(1, 2, 0) * 255
            # pred = pred.clamp(0, 255).to(torch.uint8).cpu().numpy()
            #pred = model(x).detach().squeeze(0).permute(1, 2, 0)
            pred = model(x).detach().squeeze(0)  #[12,3,512,512]
            pred = pred.clamp(0, 1).cpu().numpy()
        except RuntimeError as e:
            print(f"inference failed, error: {e}")
            continue
        
        # remove padding
        #pred = pred[:lq_resized.height, :lq_resized.width, :]
        
        
        #pred = center_expand_arr(pred, [3,311,260])  #expand后有问题
        pred = pred[0].transpose(1,2,0)  #[311,260,3]
        gt_img = gt.transpose(1,2,0)  #[311,260,3]
        lq_img = lq.transpose(1,2,0)  #[311,260,3]
        
        pred_img = (pred * 255).clip(0, 255).astype(np.uint8)
        gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)
        lq_img = (lq_img * 255).clip(0, 255).astype(np.uint8)
        
        
        
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
        
        psnr = calculate_psnr_pt(pred_2, gt_2, crop_border=0).mean()
        lpips = lpips_metric(pred_, gt_, normalize=True).mean()  #0.2679
        ssim = calculate_ssim_pt(pred_2, gt_2, crop_border=0).mean()  #0.7713
        FID = fid.calculate_fid(gt_2.transpose(0,2,3,1), pred_2.transpose(0,2,3,1), False, 1)
        musiq_metric = pyiqa.create_metric('musiq',device='cuda')
        MUSIQ = musiq_metric(pred_.unsqueeze(0)).detach().cpu().numpy()[0][0]
        niqe_metric = pyiqa.create_metric('niqe',device='cuda')
        NIQE = niqe_metric(pred_.unsqueeze(0)).detach().cpu().numpy()
        NRMSE = calculate_nrmse(pred, gt)

        psnr_all.append(np.float32(psnr))
        lpips_all.append(np.float32(lpips))
        ssim_all.append(np.float32(ssim))
        fid_all.append(np.float32(FID))
        niqe_all.append(np.float32(NIQE))
        musiq_all.append(np.float32(MUSIQ))
        nrmse_all.append(np.float32(NRMSE))

        psnr_lq = calculate_psnr_pt(lq_2, gt_2, crop_border=0).mean()
        lpips_lq = lpips_metric(lq_, gt_, normalize=True).mean()  #0.2679
        ssim_lq = calculate_ssim_pt(lq_2, gt_2, crop_border=0).mean()  #0.7713
        FID_lq = fid.calculate_fid(gt_2.transpose(0,2,3,1), lq_2.transpose(0,2,3,1), False, 1)
        musiq_metric = pyiqa.create_metric('musiq',device='cuda')
        MUSIQ_lq = musiq_metric(lq_.unsqueeze(0)).detach().cpu().numpy()[0][0]
        niqe_metric = pyiqa.create_metric('niqe',device='cuda')
        NIQE_lq = niqe_metric(lq_.unsqueeze(0)).detach().cpu().numpy()
        NRMSE_lq = calculate_nrmse(pred, lq)

        psnr_all_lq.append(np.float32(psnr_lq))
        lpips_all_lq.append(np.float32(lpips_lq))
        ssim_all_lq.append(np.float32(ssim_lq))
        fid_all_lq.append(np.float32(FID_lq))
        niqe_all_lq.append(np.float32(NIQE_lq))
        musiq_all_lq.append(np.float32(MUSIQ_lq))
        nrmse_all_lq.append(np.float32(NRMSE_lq))


        path = os.path.join(args.output, 's1', f'pred_{index}_psnr({psnr}_ssim({ssim})_lpips({lpips}.png')
        path_gt = os.path.join(args.output, 's1', f'gt_{index}.png')
        path_lq = os.path.join(args.output, 's1', f'lq_{index}_psnr({psnr_lq}_ssim({ssim_lq})_lpips({lpips_lq}.png')
        Image.fromarray(pred_img).save(path)
        Image.fromarray(gt_img).save(path_gt)
        Image.fromarray(lq_img).save(path_lq)
    
    openneuro_dict = {}
    openneuro_dict['target_grade'] = target_grades_1
    openneuro_dict['headmotion_grade'] = lq1_grades
    openneuro_dict['headmotion_psnr'] = psnr_all_lq
    openneuro_dict['headmotion_ssim'] = ssim_all_lq
    openneuro_dict['headmotion_lpips'] = lpips_all_lq
    
    openneuro_dict['pred_psnr'] = psnr_all
    openneuro_dict['pred_ssim'] = ssim_all
    openneuro_dict['pred_lpips'] = lpips_all

    openneuro_res = pd.DataFrame(openneuro_dict)
    openneuro_res.to_csv('/public_bme/data/lifeng/data/moco/test/openneuro/data/openneuro_s1_swinir.csv')

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

    #计算headmotion_2队列

    psnr_all = []
    ssim_all = []
    lpips_all = []
    fid_all = []
    niqe_all = []
    musiq_all = []
    nrmse_all = []

    psnr_all_lq = []
    ssim_all_lq = []
    lpips_all_lq = []
    fid_all_lq = []
    niqe_all_lq = []
    musiq_all_lq = []
    nrmse_all_lq = []
    
    pbar = tqdm(range(target_all_2.shape[0]))
    #for file_path in pbar:
    # for num in range(2):
    #     source_all = f'source_all_{num+1}'
    for index in pbar:
        x = source_all_2[index][2]  #(512,512)
        gt= target_all_2[index][2]

        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(int(x * args.sr_scale) for x in lq.size), Image.BICUBIC
            )
        #lq_resized = auto_resize(lq, args.image_size)
        # lq_resized = center_crop_arr(x, args.image_size) #(512,512)
        # gt = center_crop_arr(gt, args.image_size) #(512,512)
        
        lq_resized = x

        lq = np.stack((lq_resized, lq_resized, lq_resized), axis=0)
        #lq_resized = center_crop_arr_new(x, args.image_size) #(512,512)
        lq_resized = np.stack((lq_resized, lq_resized, lq_resized), axis=-1)  ##(512,512,3)
        # gt = center_crop_arr(gt, args.image_size)
        # lq = center_crop_arr(x, args.image_size)
        #lq = x
        gt = np.stack((gt, gt, gt), axis=0)
        
        """Normalize img to [0,1]
        """
        max_value, min_value = lq_resized.max(), lq_resized.min() 
        lq_resized = (lq_resized - min_value) / (max_value - min_value)

        max_value, min_value = gt.max(), gt.min() 
        gt = (gt - min_value) / (max_value - min_value)

        max_value, min_value = lq.max(), lq.min() 
        lq = (lq - min_value) / (max_value - min_value)
        # padding
        #x = pad(np.array(lq_resized), scale=64)
        x = lq_resized

        #x = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
        x = torch.tensor(x, dtype=torch.float32, device=device)
        x = x.permute(2, 0, 1).unsqueeze(0).contiguous()  #[1, 3, 512, 512]
        try:
            # pred = model(x).detach().squeeze(0).permute(1, 2, 0) * 255
            # pred = pred.clamp(0, 255).to(torch.uint8).cpu().numpy()
            #pred = model(x).detach().squeeze(0).permute(1, 2, 0)
            pred = model(x).detach().squeeze(0)  #[3,512,512]
            pred = pred.clamp(0, 1).cpu().numpy()
        except RuntimeError as e:
            print(f"inference failed, error: {e}")
            continue
        
        # remove padding
        #pred = pred[:lq_resized.height, :lq_resized.width, :]
        
        
        #pred = center_expand_arr(pred, [3,311,260])  #expand后有问题
        pred = pred.transpose(1,2,0)  #[311,260,3]
        gt_img = gt.transpose(1,2,0)  #[311,260,3]
        lq_img = lq.transpose(1,2,0)  #[311,260,3]
        
        pred_img = (pred * 255).clip(0, 255).astype(np.uint8)
        gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)
        lq_img = (lq_img * 255).clip(0, 255).astype(np.uint8)
        
        
        
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
        
        psnr = calculate_psnr_pt(pred_2, gt_2, crop_border=0).mean()
        lpips = lpips_metric(pred_, gt_, normalize=True).mean()  #0.2679
        ssim = calculate_ssim_pt(pred_2, gt_2, crop_border=0).mean()  #0.7713
        FID = fid.calculate_fid(gt_2.transpose(0,2,3,1), pred_2.transpose(0,2,3,1), False, 1)
        musiq_metric = pyiqa.create_metric('musiq',device='cuda')
        MUSIQ = musiq_metric(pred_.unsqueeze(0)).detach().cpu().numpy()[0][0]
        niqe_metric = pyiqa.create_metric('niqe',device='cuda')
        NIQE = niqe_metric(pred_.unsqueeze(0)).detach().cpu().numpy()
        NRMSE = calculate_nrmse(pred, gt)

        psnr_all.append(np.float32(psnr))
        lpips_all.append(np.float32(lpips))
        ssim_all.append(np.float32(ssim))
        fid_all.append(np.float32(FID))
        niqe_all.append(np.float32(NIQE))
        musiq_all.append(np.float32(MUSIQ))
        nrmse_all.append(np.float32(NRMSE))

        psnr_lq = calculate_psnr_pt(lq_2, gt_2, crop_border=0).mean()
        lpips_lq = lpips_metric(lq_, gt_, normalize=True).mean()  #0.2679
        ssim_lq = calculate_ssim_pt(lq_2, gt_2, crop_border=0).mean()  #0.7713
        FID_lq = fid.calculate_fid(gt_2.transpose(0,2,3,1), lq_2.transpose(0,2,3,1), False, 1)
        musiq_metric = pyiqa.create_metric('musiq',device='cuda')
        MUSIQ_lq = musiq_metric(lq_.unsqueeze(0)).detach().cpu().numpy()[0][0]
        niqe_metric = pyiqa.create_metric('niqe',device='cuda')
        NIQE_lq = niqe_metric(lq_.unsqueeze(0)).detach().cpu().numpy()
        NRMSE_lq = calculate_nrmse(pred, lq)

        psnr_all_lq.append(np.float32(psnr_lq))
        lpips_all_lq.append(np.float32(lpips_lq))
        ssim_all_lq.append(np.float32(ssim_lq))
        fid_all_lq.append(np.float32(FID_lq))
        niqe_all_lq.append(np.float32(NIQE_lq))
        musiq_all_lq.append(np.float32(MUSIQ_lq))
        nrmse_all_lq.append(np.float32(NRMSE_lq))

        path = os.path.join(args.output, 's2', f'pred_{index}_psnr({psnr}_ssim({ssim})_lpips({lpips}.png')
        path_gt = os.path.join(args.output, 's2', f'gt_{index}.png')
        path_lq = os.path.join(args.output, 's2', f'lq_{index}_psnr({psnr_lq}_ssim({ssim_lq})_lpips({lpips_lq}.png')
        Image.fromarray(pred_img).save(path)
        Image.fromarray(gt_img).save(path_gt)
        Image.fromarray(lq_img).save(path_lq)
    
    openneuro_dict = {}
    openneuro_dict['target_grade'] = target_grades_2
    openneuro_dict['headmotion_grade'] = lq2_grades
    openneuro_dict['headmotion_psnr'] = psnr_all_lq
    openneuro_dict['headmotion_ssim'] = ssim_all_lq
    openneuro_dict['headmotion_lpips'] = lpips_all_lq
    
    openneuro_dict['pred_psnr'] = psnr_all
    openneuro_dict['pred_ssim'] = ssim_all
    openneuro_dict['pred_lpips'] = lpips_all

    openneuro_res = pd.DataFrame(openneuro_dict)
    openneuro_res.to_csv('/public_bme/data/lifeng/data/moco/test/openneuro/data/openneuro_s2_swinir.csv')

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
    # fileName = '/public_bme/data/lifeng/data/moco/openneuro/sub-992238/sub-992238_scans.tsv'

    # tsv_file = pd.read_csv(
    #     fileName,
    #     sep='\t',
    #     header=0,
    #     index_col='filename'
    # )
    # print('success')
