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

def load_nii(nii_path):
    try:
        image = sitk.ReadImage(nii_path)
    except:
        assert False, f"failed to load image {nii_path}"
    image_array = sitk.GetArrayFromImage(image)
    
    return image_array

def save_nii(array, id_name):
    img = sitk.GetImageFromArray(array)
    sitk.WriteImage(img, '/public_bme/data/lifeng/data/LISA/LISA_train_motion/' + id_name)

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

class MotionbrainDataset_2(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        # crop_type: str,
        # use_hflip: bool
    ) -> "MotionbrainDataset_2":
        super(MotionbrainDataset_2, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        #root = '/public_bme/data/lifeng/data/LISA/LISA2024Task1'
        
        #self.paths = [os.path.join(file_list, i) for i in os.listdir(file_list) if i.endswith('.nii.gz') and not i.startswith('.')]
        #self.paths = os.listdir(self.file_list)

        #self.paths = [load_file_list(file_list)[0],load_file_list(file_list)[1]]
        self.out_size = out_size       #256
        # self.crop_type = crop_type     #center
        # assert self.crop_type in ["none", "center", "random"]   
        # self.use_hflip = use_hflip
        self.target_all = []
        self.source_all = []
        self.num = 0
        # degradation configurations
        # self.blur_kernel_size = blur_kernel_size
        # self.kernel_list = kernel_list
        # self.kernel_prob = kernel_prob
        # self.blur_sigma = blur_sigma
        # self.downsample_range = downsample_range
        # self.noise_range = noise_range
        # self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        
        gt_path = self.paths[index]
        #gt_path = '/public_bme/data/lifeng/data/LISA/LISA2024Task1/LISA_0001_LF_axi.nii.gz'
        
        #id_name = gt_path.split('/')[-1]
        success = False
        try:
            pil_img_3D = load_nii(gt_path)
            #pil_img_3D = np.transpose(pil_img_3D,(0,2,1))  
            #pil_img_3D = load_nii(gt_path.replace('T1w','T2w'))
            success = True
        except:
            return self.__getitem__(index+1)
        assert success, f"failed to load image {gt_path}"
        
        #pil_img_3D = center_pad_arr(pil_img_3D, 3, self.out_size)  #[320,320,320]
        #pil_img_3D = center_resize_arr(pil_img_3D, 256)
        pil_img_3D = center_crop_arr(pil_img_3D, 512)
        length = pil_img_3D.shape[0]
        median = round(length / 2)

        for _ in range(5):
            index_A = random.randint(median-40, median+50)

            # if index % 50 == 0 and index != 0:
            #     np.save(f'/public_bme/data/lifeng/data/hcp_train_gt_{self.num}_3d.npy', self.target_all)
            #     np.save(f'/public_bme/data/lifeng/data/hcp_train_lq_{self.num}_3d.npy', self.source_all)

            #     self.num += 1
            


            #for index_A in range(median-40, median+50):
            
            # pil_img_3D = np.transpose(pil_img_3D,(0,2,1))
            # img = pil_img_3D[np.newaxis,:,:,:]
            
            # ------------------------ generate lq image ------------------------ #
                # dis = random.randint(0, 1) #2
                # degree = random.randint(0, 1)  #(1,6)
                # steps = random.randint(1, 15)

            ##往前移了一个
            corruption_scheme = ['piecewise_constant', 'gaussian', 'piecewise_transient'] 
            selected_corruption_scheme = np.random.choice(corruption_scheme)
            #steps = 30 #2
            """Generate the motion artifacts and return the img and its motion arguments
            """
            # Motion = tio.RandomMotion(degrees=degree, translation=dis,num_transforms=steps) #5,0,2  | 8,1,2
            # #transform = tio.Compose([Motion])
            # transformed_img = Motion(img) #
            # transformed_img_3D = transformed_img[0,:,:,:]


            # ------------------------ generate lq image ------------------------ #
            try:
                ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[15, 25],
                                                    corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                                    n_seg=8)
                transformed_img_3D = torch.tensor(ms_layer.layer_op(pil_img_3D))
                #transformed_img_3D, transformed_img_3D_T2 = ms_layer.layer_op(pil_img_3D,pil_img_3D_T2)
            except:
                ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[30, 30],
                                                    corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                                    n_seg=8)
                transformed_img_3D = torch.tensor(ms_layer.layer_op(pil_img_3D))
                #transformed_img_3D, transformed_img_3D_T2 = ms_layer.layer_op(pil_img_3D,pil_img_3D_T2)
            transformed_img_3D = normalize_mri_volume(transformed_img_3D)
            max_value, min_value = transformed_img_3D.max(), transformed_img_3D.min() 
            transformed_img_3D = (transformed_img_3D - min_value) / (max_value - min_value)

            pil_img_3D = normalize_mri_volume(pil_img_3D)
            max_value, min_value = pil_img_3D.max(), pil_img_3D.min() 
            pil_img_3D = (pil_img_3D - min_value) / (max_value - min_value)

                
            try:
                pil_img = pil_img_3D[index_A]
                lq_img = transformed_img_3D[index_A]
                #pil_img_T2 = pil_img_3D_T2[index_A]
                # pil_img = pil_img_3D[index_A-2:index_A+4]
                # lq_img = transformed_img_3D[index_A-2:index_A+4]
                
            except:
                print(f"failed to load image {gt_path}")

                index_A = random.randint(90, 130)
                
                pil_img = pil_img_3D[index_A]
                # #pil_img_T2 = pil_img_3D_T2[index_A]

                lq_img = transformed_img_3D[index_A]

            pil_img = torch.flip(pil_img, dims=[0,1]) #[6,320,320], #dims=[0,1]
            lq_img = torch.flip(lq_img, dims=[0,1])

            source = np.array(lq_img).astype(np.float32)
            target = np.array(pil_img).astype(np.float32)
            self.source_all.append(source)
            self.target_all.append(target)
        
    def __len__(self) -> int:
        return len(self.paths)

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
    np.save('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_t1_10-25_hq.npy', clean_all)
    np.save('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_t1_10-25_lq.npy', corrupt_all)

    # gt_3D = load_nii('/public_bme/data/lifeng/data/moco/test/sub-000149_acq-standard_T1w.nii.gz')
    # gt_3D = np.transpose(gt_3D,(0,2,1))
    # max_value, min_value = gt_3D.max(), gt_3D.min() 
    # gt_3D = (gt_3D - min_value) / (max_value - min_value)
    # lq_3D = load_nii('/public_bme/data/lifeng/data/moco/test/20160328-ST002-Elison_BSLERP_505499_03_01_MR-SE002-T1w.nii.gz')
    # lq_3D = np.transpose(lq_3D,(0,2,1))
    # max_value, min_value = lq_3D.max(), lq_3D.min() 
    # lq_3D = (lq_3D - min_value) / (max_value - min_value)
    # selected_slices = []
    # for i in range(5):
    #     index_A = random.randint(120, 160)
    #     selected_slices.append(index_A)
    # for slice in selected_slices:
    #     # gt_img = (gt_3D[slice] * 255.0).astype('uint8')
    #     # imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/{slice}_hq.jpg', gt_img)
    #     lq_img = (lq_3D[:,:,slice] * 255.0).astype('uint8')
    #     imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/{slice}_lq.jpg', lq_img)