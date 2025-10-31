from typing import Sequence, Dict, Union
import math
import time
import random
import SimpleITK as sitk
import torchio as tio
import sys
import imageio
#sys.path.append("/home_data/home/lifeng2023/code/moco/DiffBIR-main")
sys.path.append("/public_bme/data/lifeng/code/moco/TS_BHIR")
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from einops import rearrange
from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)
import torch
from dataset import motion_sim

def load_nii(nii_path):
    try:
        image = sitk.ReadImage(nii_path)
    except:
        assert False, f"failed to load image {nii_path}"
    image_array = sitk.GetArrayFromImage(image)
    
    return image_array

class MotionbrainDataset_ms(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool
    ) -> "MotionbrainDataset_ms":
        super(MotionbrainDataset_ms, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        #self.paths = [load_file_list(file_list)[0],load_file_list(file_list)[1]]
        self.num_frame = 5
        self.out_size = out_size       #256
        self.crop_type = crop_type     #center
        assert self.crop_type in ["none", "center", "random"]   
        self.use_hflip = use_hflip
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
        
        success = False
        try:
            pil_img_3D = load_nii(gt_path)
            success = True
        except:
            time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        
        
        # pil_img_3D = np.transpose(pil_img_3D,(0,2,1))
        # img = pil_img_3D[np.newaxis,:,:,:]
        
        # ------------------------ generate lq image ------------------------ #
        # dis = random.randint(0, 1) #2
        # degree = random.randint(0, 1)  #(1,6)
        # steps = random.randint(1, 15)
        corruption_scheme = ['gaussion', 'piecewise_transient', 'piecewise_constant']
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
            ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[0, 40],
                                                corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                                n_seg=8)
            transformed_img_3D = ms_layer.layer_op(pil_img_3D)
        except:
            ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[30, 30],
                                                corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
                                                n_seg=8)
            transformed_img_3D = ms_layer.layer_op(pil_img_3D)

        
        index = gt_path.split('/')[-2]

        length = pil_img_3D.shape[0]
        median = round(length / 2)
        index_A = random.randint(median-40, median+50)

        pil_img = pil_img_3D[index_A+2:index_A+7]

        lq_img = transformed_img_3D[index_A:index_A+10]


        if self.crop_type == "center":
            img_gt = center_crop_arr(pil_img, self.out_size)
            lq_img = center_crop_arr(lq_img, self.out_size)

        elif self.crop_type == "random":
            img_gt = random_crop_arr(pil_img, self.out_size)
            lq_img = random_crop_arr(lq_img, self.out_size)

        else:
            img_gt = np.array(pil_img)
            lq_img = np.array(lq_img)

            assert img_gt.shape[-2,-1] == (self.out_size, self.out_size)
            assert lq_img.shape[-2,-1] == (self.out_size, self.out_size)
        
        img_gt = np.stack((img_gt, img_gt, img_gt), axis=3)
        lq_img = np.stack((lq_img, lq_img, lq_img), axis=3)
        """只取1层
        """
        #img_gt = img_gt[2]
        """Normalize img to [0,1]
        """
        max_value, min_value = img_gt.max(), img_gt.min() 
        img_gt = (img_gt - min_value) / (max_value - min_value)

        # max_value, min_value = img_gt_plus1.max(), img_gt_plus1.min() 
        # img_gt_plus1 = (img_gt_plus1 - min_value) / (max_value - min_value)

        # max_value, min_value = img_gt_minus1.max(), img_gt_minus1.min() 
        # img_gt_minus1 = (img_gt_minus1 - min_value) / (max_value - min_value)
        """Normalize img to [0,1]
        """
        max_value, min_value = lq_img.max(), lq_img.min() 
        lq_img = (lq_img - min_value) / (max_value - min_value)

        # max_value, min_value = lq_img_plus1.max(), lq_img_plus1.min() 
        # lq_img_plus1 = (lq_img_plus1 - min_value) / (max_value - min_value)

        # max_value, min_value = lq_img_minus1.max(), lq_img_minus1.min() 
        # lq_img_minus1 = (lq_img_minus1 - min_value) / (max_value - min_value)
        #[-1, 1]
        target = (img_gt * 2 - 1).astype(np.float32)
        # target_plus1 = (img_gt_plus1 * 2 - 1).astype(np.float32)
        # target_minus1 = (img_gt_minus1 * 2 - 1).astype(np.float32)
        
        #[0, 1]
        source = lq_img.astype(np.float32)
        # source_plus1 = lq_img_plus1.astype(np.float32)
        # source_minus1 = lq_img_minus1.astype(np.float32)
        
        # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
        # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
        # # # 保存图片到本地
        # imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/{index}_{index_A}_hq.jpg', new_im_hq)
        # data = (source * 255.0).astype('uint8')  # 转换数据类型
        # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
        # # # # 保存图片到本地
        # imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/{index}_{index_A}_lq.jpg', new_im_lq)
        # dicom_itk_lq = sitk.GetImageFromArray(lq_img)
        # sitk.WriteImage(dicom_itk_lq,f'/public_bme/data/lifeng/data/moco/test/{index}_lq.nii.gz')
        # dicom_itk_hq = sitk.GetImageFromArray(img_gt)
        # sitk.WriteImage(dicom_itk_hq,f'/public_bme/data/lifeng/data/moco/test/{index}__hq.nii.gz')
        
        return dict(jpg=target, txt="", hint=source)#,dict(jpg=target_plus1, txt="", hint=source_plus1),dict(jpg=target_minus1, txt="", hint=source_minus1)

    def __len__(self) -> int:
        return len(self.paths)

if __name__ == '__main__':
    data = MotionbrainDataset_ms(
        file_list='/public_bme/data/lifeng/data/val_hcp.list',
        out_size = 512,
        crop_type= 'center',
        #crop_type= 'none',
        use_hflip=False
        )
    for i in data:
       pass
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