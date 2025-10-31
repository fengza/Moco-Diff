from typing import Sequence, Dict, Union
import math
import time
import random
import SimpleITK as sitk
import torchio as tio
import sys
import os
import imageio
#sys.path.append("/home_data/home/lifeng2023/code/moco/DiffBIR-main")
sys.path.append("/public_bme/data/lifeng/code/moco/TS_BHIR")
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from einops import rearrange
from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr, center_pad_arr,center_resize_arr
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)
import torch
from dataset import motion_sim
import torch.nn.functional as F
import pandas as pd

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

def load_nii(nii_path):
    try:
        image = sitk.ReadImage(nii_path)
    except:
        assert False, f"failed to load image {nii_path}"
    image_array = sitk.GetArrayFromImage(image)
    
    return image_array

def sim_motion(pil_img_3D):
    corruption_scheme = ['piecewise_constant','gaussion', 'piecewise_transient'] #
    selected_corruption_scheme = np.random.choice(corruption_scheme)
    
    ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[15, 30],
                                        corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                        n_seg=8)
    lq_img_3D = ms_layer.layer_op(pil_img_3D)

    return lq_img_3D

class MotionbrainDataset_2(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        label_path: str,
        # crop_type: str,
        # use_hflip: bool
    ) -> "MotionbrainDataset_2":
        super(MotionbrainDataset_2, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
       
        #self.label = pd.read_excel(label_path)

        #self.paths = [load_file_list(file_list)[0],load_file_list(file_list)[1]]
        self.out_size = out_size       #256
        # self.crop_type = crop_type     #center
        # assert self.crop_type in ["none", "center", "random"]   
        # self.use_hflip = use_hflip
        # self.target_all = []
        # self.source_all = []
        # self.num = 0
        # degradation configurations
        # self.blur_kernel_size = blur_kernel_size
        # self.kernel_list = kernel_list
        # self.kernel_prob = kernel_prob
        # self.blur_sigma = blur_sigma
        # self.downsample_range = downsample_range
        # self.noise_range = noise_range
        # self.jpeg_range = jpeg_range

    # def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
    #     # load gt image
    #     # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
    #     id = str(self.label.iloc[index]['ID']) + '.jpg'
    #     lq_path = os.path.join(self.file_list, id)
        
    #     label = int(self.label.iloc[index]['label'])
        
    #     lq_img = imageio.imread(lq_path, pilmode='L')
    #     lq_img = np.array(lq_img)
    #     lq_img = center_crop_arr(lq_img, self.out_size)
        
    #     lq_img = np.stack((lq_img, lq_img, lq_img), axis=-1)

    #     """Normalize img to [0,1]
    #     """
    #     max_value, min_value = lq_img.max(), lq_img.min() 
    #     lq_img = (lq_img - min_value) / (max_value - min_value)
        
    #     # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
    #     # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
    #     # # # 保存图片到本地
    #     # imageio.imsave(f'/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/{index}_{index_A}_hq.jpg', new_im_hq)
    #     # data = (source * 255.0).astype('uint8')  # 转换数据类型
    #     # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
    #     # # # # 保存图片到本地
    #     # imageio.imsave(f'/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/{index}_{index_A}_lq.jpg', new_im_lq)
    #     # dicom_itk_lq = sitk.GetImageFromArray(lq_img)
    #     # sitk.WriteImage(dicom_itk_lq,f'/public_bme/data/lifeng/data/moco/test/{index}_lq.nii.gz')
    #     # dicom_itk_hq = sitk.GetImageFromArray(img_gt)
    #     # sitk.WriteImage(dicom_itk_hq,f'/public_bme/data/lifeng/data/moco/test/{index}__hq.nii.gz')
        
    #     return dict(jpg=lq_img, txt="", hint=label)#,dict(jpg=target_plus1, txt="", hint=source_plus1),dict(jpg=target_minus1, txt="", hint=source_minus1)

    def __getitem__(self, index):
        gt_path = self.paths[index]
        success = False
        try:
            pil_img_3D = load_nii(gt_path)
            #pil_img_3D = np.transpose(pil_img_3D,(0,2,1))  
            #pil_img_3D = load_nii(gt_path.replace('T1w','T2w'))
            success = True
        except:
            return self.__getitem__(index+1)
        assert success, f"failed to load image {gt_path}"

        transformed_img_3D = sim_motion(pil_img_3D)
        
        length = pil_img_3D.shape[0]
        median = round(length / 2)
        index_A = random.randint(median-40, median+50)

        img_gt = pil_img_3D[index_A-2:index_A+4]
        lq_single = transformed_img_3D[index_A-2:index_A+4]
        img_gt = center_crop_arr(img_gt, 512, dim=3)
        lq_single = center_crop_arr(lq_single, 512, dim=3)

        img_gt = img_gt[2]
        lq_single = lq_single[2]
        lq_img = np.stack((lq_single, lq_single, lq_single), axis=-1)
        img_gt = np.stack((img_gt, img_gt, img_gt), axis=-1)

        max_value, min_value = img_gt.max(), img_gt.min() 
        img_gt = (img_gt - min_value) / (max_value - min_value)

        max_value, min_value = lq_img.max(), lq_img.min() 
        lq_img = (lq_img - min_value) / (max_value - min_value)
        
        # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
        # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
        # # # 保存图片到本地
        # imageio.imsave(f'/public_bme/data/lifeng/data/moco/mic_extension/test/{index}_hq.jpg', new_im_hq)
        # data = (lq_img * 255.0).astype('uint8')  # 转换数据类型
        # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
        # # # # 保存图片到本地
        # imageio.imsave(f'/public_bme/data/lifeng/data/moco/mic_extension/test/{index}_lq.jpg', new_im_lq)

        target = (img_gt * 2 - 1).astype(np.float32) #target.astype(np.float32)#
        #self.target_all.append(target)
        
        #[0, 1]
        source = lq_img.astype(np.float32)
        return dict(jpg=target, txt="", hint=source)


    def __len__(self) -> int:
        return self.label.shape[0]

if __name__ == '__main__':
    data = MotionbrainDataset_2(
        file_list= '/public_bme/data/lifeng/data/val_hcp.list', #'/public_bme/data/lifeng/data/LISA/LISA2024Task1', #'/public_bme/data/lifeng/data/val_hcp.list',
        out_size = 512,
        # crop_type= 'center',
        # #crop_type= 'none',
        # use_hflip=False
        )
    for i in data:
       pass
    clean_all = np.array(data.target_all)
    corrupt_all = np.array(data.source_all)
    np.save('/public_bme/data/lifeng/data/moco/test/t2/random/hcp_val_t1_0-1.npy', clean_all)
    np.save('/public_bme/data/lifeng/data/moco/test/t2/random/hcp_val_t1_0-1.npy', corrupt_all)
