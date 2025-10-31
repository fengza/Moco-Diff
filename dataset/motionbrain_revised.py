from typing import Sequence, Dict, Union
import math
import time
import random
import SimpleITK as sitk
import warnings
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

warnings.filterwarnings('ignore')
reader = sitk.ImageSeriesReader()

def load_dcm(folder_path, error_ids=[]):
    try:
        dicomFiles = reader.GetGDCMSeriesFileNames(folder_path)
        reader.SetFileNames(dicomFiles)
        vol = reader.Execute()
    except:
        error_ids.append(folder_path.split('/')[-1])
        return None
    image_array = sitk.GetArrayFromImage(vol)
    return image_array

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
        #self.paths = load_file_list(file_list)[0:20]
        #root = '/public_bme/data/lifeng/data/LISA/LISA2024Task1'
        
        #self.paths = [os.path.join(file_list, i) for i in os.listdir(file_list) if i.endswith('.nii.gz') and not i.startswith('.')][-20:]
        self.paths = ['/public_bme/data/lifeng/data/moco_multi/ccbd_validation/T1/CCBD_eymrae_wjh_CCBD_eymrae_111446/CCBD_T1ISO_acsFSP_NDC_301']
        #self.paths = os.listdir(self.file_list)

        #self.paths = [load_file_list(file_list)[0],load_file_list(file_list)[1]]
        self.out_size = out_size       #256
        # self.crop_type = crop_type     #center
        # assert self.crop_type in ["none", "center", "random"]   
        # self.use_hflip = use_hflip
        self.target_all = []
        self.source_all = []
        self.num = 1
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
        
        #gt_path = self.paths[index]
        gt_path = '/public_bme/data/lifeng/data/moco_multi/ccbd_validation/T2/CCBD_ujiatw_CCBD_ujiatw_202143/CCBD_T2ISO_acsMX_NDC_501'
        pil_img_3D = load_dcm(gt_path)
        pil_img_3D = np.transpose(pil_img_3D,(1,0,2))  
        # success = False
        # try:
        #     pil_img_3D = load_nii(gt_path)
        #     #pil_img_3D = np.transpose(pil_img_3D,(0,2,1))  
        #     pil_img_3D_T2 = load_nii(gt_path.replace('T1','T2'))
        #     pil_img_3D_PD = load_nii(gt_path.replace('T1','PD'))
        #     success = True
        # except:
        #     return self.__getitem__(index+1)
        # assert success, f"failed to load image {gt_path}"
        
        #pil_img_3D = center_pad_arr(pil_img_3D, 3, self.out_size)  #[320,320,320]
        #pil_img_3D = center_resize_arr(pil_img_3D, 256)
        #pil_img_3D = center_crop_arr(pil_img_3D, 512)
        # length = pil_img_3D.shape[0]
        # median = round(length / 2)
        for index_A in range(74,84):
            #index_A = random.randint(median-40, median+50)

            #if index % 50 == 0 and index != 0:
                # np.save(f'/public_bme/data/lifeng/data/hcp_train_gt_{self.num}_3d.npy', self.target_all)
                # np.save(f'/public_bme/data/lifeng/data/hcp_train_lq_{self.num}_3d.npy', self.source_all)

                # self.num += 1
            
                # self.target_all = []
                # self.source_all = []

            #for index_A in range(median-40, median+50):
            
            # pil_img_3D = np.transpose(pil_img_3D,(0,2,1))
            # img = pil_img_3D[np.newaxis,:,:,:]
            
            # ------------------------ generate lq image ------------------------ #
                # dis = random.randint(0, 1) #2
                # degree = random.randint(0, 1)  #(1,6)
                # steps = random.randint(1, 15)

            # ##往前移了一个
            # corruption_scheme = ['piecewise_constant', 'gaussian', 'piecewise_transient'] 
            # selected_corruption_scheme = np.random.choice(corruption_scheme)
            # #steps = 30 #2
            # """Generate the motion artifacts and return the img and its motion arguments
            # """
            # # Motion = tio.RandomMotion(degrees=degree, translation=dis,num_transforms=steps) #5,0,2  | 8,1,2
            # # #transform = tio.Compose([Motion])
            # # transformed_img = Motion(img) #
            # # transformed_img_3D = transformed_img[0,:,:,:]


            # # ------------------------ generate lq image ------------------------ #
            # try:
            #     ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[30, 40],
            #                                         corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
            #                                         n_seg=8)
            #     #transformed_img_3D = torch.tensor(ms_layer.layer_op(pil_img_3D))
            #     transformed_img_3D = ms_layer.layer_op(pil_img_3D)
            #     transformed_img_3D_T2 = ms_layer.layer_op(pil_img_3D_T2)
            #     transformed_img_3D_PD = ms_layer.layer_op(pil_img_3D_PD)
            #     #transformed_img_3D, transformed_img_3D_T2 = ms_layer.layer_op(pil_img_3D,pil_img_3D_T2)
            # except:
            #     ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[30, 30],
            #                                         corruption_scheme=selected_corruption_scheme,#'piecewise_transient',
            #                                         n_seg=8)
            #     transformed_img_3D = ms_layer.layer_op(pil_img_3D)
            #     transformed_img_3D_T2 = ms_layer.layer_op(pil_img_3D_T2)
            #     transformed_img_3D_PD = ms_layer.layer_op(pil_img_3D_PD)
                #transformed_img_3D, transformed_img_3D_T2 = ms_layer.layer_op(pil_img_3D,pil_img_3D_T2)
            # transformed_img_3D = normalize_mri_volume(transformed_img_3D)
            # max_value, min_value = transformed_img_3D.max(), transformed_img_3D.min() 
            # transformed_img_3D = (transformed_img_3D - min_value) / (max_value - min_value)

            # pil_img_3D = normalize_mri_volume(pil_img_3D)
            # max_value, min_value = pil_img_3D.max(), pil_img_3D.min() 
            # pil_img_3D = (pil_img_3D - min_value) / (max_value - min_value)
            #save_nii(transformed_img_3D, id_name)
            # img_gt = np.array(pil_img_3D)
            # lq_img = np.array(transformed_img_3D)

            # img_gt_T2 = np.array(pil_img_3D_T2)
            # lq_img_T2 = np.array(transformed_img_3D_T2)
                
            try:
                pil_img = np.rot90(pil_img_3D[index_A],k=3) #k=3
                # lq_img = transformed_img_3D[index_A]
                # pil_img_T2 = pil_img_3D_T2[index_A]
                # lq_img_T2 = transformed_img_3D_T2[index_A]
                # pil_img_PD = pil_img_3D_PD[index_A]
                # lq_img_PD = transformed_img_3D_PD[index_A]

                # pil_img = pil_img_3D[index_A-2:index_A+4]
                # lq_img = transformed_img_3D[index_A-2:index_A+4]
                
            except:
                print(f"failed to load image {gt_path}")

                index_A = random.randint(90, 130)
                
                pil_img = np.rot90(pil_img_3D[index_A])
                # lq_img = transformed_img_3D[index_A]
                # pil_img_T2 = pil_img_3D_T2[index_A]
                # lq_img_T2 = transformed_img_3D_T2[index_A]
                # pil_img_PD = pil_img_3D_PD[index_A]
                # lq_img_PD = transformed_img_3D_PD[index_A]
                # pil_img = pil_img_3D[index_A-2:index_A+4]
                # lq_img = transformed_img_3D[index_A-2:index_A+4]
            
            # pil_img = pil_img_3D[index_A-2:index_A+4]
            # lq_img = transformed_img_3D[index_A-2:index_A+4]

            # if self.crop_type == "center":
            # pil_img = center_crop_arr(np.array(pil_img), self.out_size)
            # lq_img = center_crop_arr(np.array(lq_img), self.out_size)
            #img_gt = center_pad_arr(pil_img, 2, 512)
            #lq_img = center_pad_arr(lq_img, 2, 512)
            #     # img_gt = center_crop_arr(pil_img_3D, self.out_size)
            #     # lq_img = center_crop_arr(transformed_img_3D, self.out_size)
                
            #     # img_gt_plus1 = center_crop_arr(pil_img_plus1, self.out_size)
            #     # lq_img_plus1 = center_crop_arr(lq_img_plus1, self.out_size)
                
            #     # img_gt_minus1 = center_crop_arr(pil_img_minus1, self.out_size)
            #     # lq_img_minus1 = center_crop_arr(lq_img_minus1, self.out_size)
            # elif self.crop_type == "random":
            #     img_gt = random_crop_arr(pil_img, self.out_size)
            #     lq_img = random_crop_arr(lq_img, self.out_size)
            #     # img_gt = random_crop_arr(pil_img_3D, self.out_size)
            #     # lq_img = random_crop_arr(transformed_img_3D, self.out_size)
                
            #     # img_gt_plus1 = center_crop_arr(pil_img_plus1, self.out_size)
            #     # lq_img_plus1 = center_crop_arr(lq_img_plus1, self.out_size)
                
            #     # img_gt_minus1 = center_crop_arr(pil_img_minus1, self.out_size)
            #     # lq_img_minus1 = center_crop_arr(lq_img_minus1, self.out_size)
            # else:
            #     img_gt = np.array(pil_img)
            #     lq_img_T2 = np.array(lq_img_T2)
            #     # img_gt = np.array(pil_img_3D)
            #     # lq_img = np.array(transformed_img_3D)
            #     # img_gt_plus1 = np.array(pil_img_plus1)
            #     # lq_img_plus1 = np.array(lq_img_plus1)
            #     # img_gt_minus1 = np.array(pil_img_minus1)
            #     # lq_img_minus1 = np.array(lq_img_minus1)
            #     assert img_gt.shape == (self.out_size, self.out_size)
            #     assert lq_img.shape == (self.out_size, self.out_size)
            
            # pil_img = np.array(torch.flip(pil_img, dims=[1,2])) #[6,320,320], #dims=[0,1]
            # lq_img = np.array(torch.flip(lq_img, dims=[1,2]))
            
            # pil_img = np.stack((pil_img, pil_img, pil_img), axis=-1)
            # #img_gt = np.stack((img_gt, img_gt, img_gt), axis=2)
            # lq_img = np.stack((lq_img, lq_img, lq_img), axis=-1)
            
            img_gt = np.array(pil_img)
            # lq_img = np.array(lq_img)

            # img_gt_T2 = np.array(pil_img_T2)
            # lq_img_T2 = np.array(lq_img_T2)

            # img_gt_PD = np.array(pil_img_PD)
            # lq_img_PD = np.array(lq_img_PD)

                # """Normalize img to [0,1]
                # """
            # max_value, min_value = lq_img.max(), lq_img.min() 
            # lq_img = (lq_img - min_value) / (max_value - min_value)

            # max_value, min_value = img_gt.max(), img_gt.min() 
            # img_gt = (img_gt - min_value) / (max_value - min_value)

                # max_value, min_value = lq_img_T2.max(), lq_img_T2.min() 
                # lq_img_T2 = (lq_img_T2 - min_value) / (max_value - min_value)

                # max_value, min_value = img_gt_T2.max(), img_gt_T2.min() 
                # img_gt_T2 = (img_gt_T2 - min_value) / (max_value - min_value)


                # target_plus1 = (img_gt_plus1 * 2 - 1).astype(np.float32)
                # target_minus1 = (img_gt_minus1 * 2 - 1).astype(np.float32)
                
                # #[0, 1]
                # source = lq_img.astype(np.float32)
                # target = img_gt.astype(np.float32)
                
            # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
            # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
            # # # 保存图片到本地
            # imageio.imsave(f'/public_bme/data/lifeng/data/moco_multi/ccbd_validation/T2/{index_A}_lq.jpg', new_im_hq)
            # data = (lq_img[2] * 255.0).astype('uint8')  # 转换数据类型
            # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
            # # # # 保存图片到本地
            # imageio.imsave(f'/public_bme/data/lifeng/data/moco/{index}_{index_A}_lq.jpg', new_im_lq)

                # data = (img_gt_T2 * 255.0).astype('uint8')  # 转换数据类型
                # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
                # # # 保存图片到本地
                # imageio.imsave(f'/public_bme/data/lifeng/data/moco/{index}_{index_A}_hq_T2.jpg', new_im_hq)
                # data = (lq_img_T2 * 255.0).astype('uint8')  # 转换数据类型
                # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
                # # # # 保存图片到本地
                # imageio.imsave(f'/public_bme/data/lifeng/data/moco/{index}_{index_A}_lq_T2.jpg', new_im_lq)

            # target = np.stack((img_gt, img_gt_T2, img_gt_PD), axis=0)
            # source = np.stack((lq_img, lq_img_T2, lq_img_PD), axis=0)


            #target = (img_gt * 2 - 1).astype(np.float32) #target.astype(np.float32)#
            self.target_all.append(img_gt)
            
            #[0, 1]
            #target = img_gt.astype(np.float32)
            #source = lq_img.astype(np.float32)
            #self.source_all.append(source)

        self.target_all = np.array(self.target_all)
        #self.source_all = np.array(self.source_all)
        #np.save(f'/public_bme/data/lifeng/data/moco_multi/ixi_validation/3/hcp_test_gt_{self.num}_3d.npy', self.target_all)
        np.save(f'/public_bme/data/lifeng/data/moco_multi/ccbd_validation/T2/ccbd_real_lq_{self.num}_3d.npy', self.source_all)    
        self.num += 1
        self.target_all = []
        #self.source_all = []
            # # random horizontal flip
            # img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
            # h, w, _ = img_gt.shape

            # lq_img = augment(lq_img, hflip=self.use_hflip, rotation=False, return_status=False)
            # h, w, _ = img_gt.shape
            
            # """Normalize img to [0,1]
            # """
            # max_value, min_value = img_gt.max(), img_gt.min() 
            # img_gt = (img_gt - min_value) / (max_value - min_value)

            # # max_value, min_value = img_gt_plus1.max(), img_gt_plus1.min() 
            # # img_gt_plus1 = (img_gt_plus1 - min_value) / (max_value - min_value)

            # # max_value, min_value = img_gt_minus1.max(), img_gt_minus1.min() 
            # # img_gt_minus1 = (img_gt_minus1 - min_value) / (max_value - min_value)
            # """Normalize img to [0,1]
            # """
            # max_value, min_value = lq_img.max(), lq_img.min() 
            # lq_img = (lq_img - min_value) / (max_value - min_value)

            # # max_value, min_value = lq_img_plus1.max(), lq_img_plus1.min() 
            # # lq_img_plus1 = (lq_img_plus1 - min_value) / (max_value - min_value)

            # # max_value, min_value = lq_img_minus1.max(), lq_img_minus1.min() 
            # # lq_img_minus1 = (lq_img_minus1 - min_value) / (max_value - min_value)
            # #[-1, 1]
            # target = (img_gt * 2 - 1).astype(np.float32)
            # # target_plus1 = (img_gt_plus1 * 2 - 1).astype(np.float32)
            # # target_minus1 = (img_gt_minus1 * 2 - 1).astype(np.float32)
            
            # #[0, 1]
            # source = lq_img.astype(np.float32)
            # source_plus1 = lq_img_plus1.astype(np.float32)
            # source_minus1 = lq_img_minus1.astype(np.float32)
            
            # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
            # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
            # # # 保存图片到本地
            # imageio.imsave(f'/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/{index}_{index_A}_hq.jpg', new_im_hq)
            # data = (source * 255.0).astype('uint8')  # 转换数据类型
            # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
            # # # # 保存图片到本地
            # imageio.imsave(f'/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/script_imgs/{index}_{index_A}_lq.jpg', new_im_lq)
            # dicom_itk_lq = sitk.GetImageFromArray(lq_img)
            # sitk.WriteImage(dicom_itk_lq,f'/public_bme/data/lifeng/data/moco/test/{index}_lq.nii.gz')
            # dicom_itk_hq = sitk.GetImageFromArray(img_gt)
            # sitk.WriteImage(dicom_itk_hq,f'/public_bme/data/lifeng/data/moco/test/{index}__hq.nii.gz')
            
            #return dict(jpg=target, txt="", hint=source)#,dict(jpg=target_plus1, txt="", hint=source_plus1),dict(jpg=target_minus1, txt="", hint=source_minus1)

    # def __getitem__(self, index):
    #     gt_path = os.path.join(self.file_list, self.paths[index])
    #     lq_path = gt_path.replace('hq','lq')
 
    #     img_gt = imageio.imread(gt_path, pilmode='L')
    #     lq_img = imageio.imread(lq_path, pilmode='L')
    #     img_gt = np.array(img_gt)
    #     lq_img = np.array(lq_img)
    #     img_gt = F.interpolate(torch.Tensor(img_gt.astype("float32")).unsqueeze(0).unsqueeze(0),(512,512), mode='bicubic').squeeze(0).squeeze(0)
    #     lq_img = F.interpolate(torch.Tensor(lq_img.astype("float32")).unsqueeze(0).unsqueeze(0),(512,512), mode='bicubic').squeeze(0).squeeze(0)

    #     img_gt = np.array(img_gt)
    #     lq_img = np.array(lq_img)

    #     img_gt = np.stack((img_gt, img_gt, img_gt), axis=2)
    #     lq_img = np.stack((lq_img, lq_img, lq_img), axis=2)
        
    #     """Normalize img to [0,1]
    #     """
    #     max_value, min_value = img_gt.max(), img_gt.min() 
    #     img_gt = (img_gt - min_value) / (max_value - min_value)

    #     """Normalize img to [0,1]
    #     """
    #     max_value, min_value = lq_img.max(), lq_img.min() 
    #     lq_img = (lq_img - min_value) / (max_value - min_value)

    #     source = lq_img.astype(np.float32)
    #     target = img_gt.astype(np.float32)
        
    #     # data = (img_gt * 255.0).astype('uint8')  # 转换数据类型
    #     # new_im_hq = Image.fromarray(data)  # 调用Image库，数组归一化
    #     # # # 保存图片到本地
    #     # imageio.imsave(f'/public_bme/data/lifeng/data/moco/mic_extension/test/{index}_hq.jpg', new_im_hq)
    #     # data = (lq_img * 255.0).astype('uint8')  # 转换数据类型
    #     # new_im_lq = Image.fromarray(data)  # 调用Image库，数组归一化
    #     # # # # 保存图片到本地
    #     # imageio.imsave(f'/public_bme/data/lifeng/data/moco/mic_extension/test/{index}_lq.jpg', new_im_lq)

    #     target = (img_gt * 2 - 1).astype(np.float32) #target.astype(np.float32)#
    #     #self.target_all.append(target)
        
    #     #[0, 1]
    #     source = lq_img.astype(np.float32)
    #     return dict(jpg=target, txt="", hint=source)


    def __len__(self) -> int:
        return len(self.paths)

if __name__ == '__main__':
    data = MotionbrainDataset_2(
        file_list= '/public_bme/data/lifeng/data/IXI/T1_brain', #'/public_bme/data/lifeng/data/LISA/LISA2024Task1', #'/public_bme/data/lifeng/data/val_hcp.list',
        out_size = 512,
        # crop_type= 'center',
        # #crop_type= 'none',
        # use_hflip=False
        )
    for i in data:
       pass
    # clean_all = np.array(data.target_all)
    # corrupt_all = np.array(data.source_all)
    # np.save('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_05_hq.npy', clean_all)
    # np.save('/public_bme2/bme-wangqian2/lifeng2023/data/mic_extension/validation_set/hcp_val_05_lq.npy', corrupt_all)

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