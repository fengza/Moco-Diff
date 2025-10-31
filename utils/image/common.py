import random
import math

from PIL import Image
import numpy as np
import cv2
import torch
from torch.nn import functional as F



# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size, dim = 2):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    #while min(*pil_image.size) >= 2 * image_size:
    # while min(*pil_image.shape) >= 2 * image_size:
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.shape), resample=Image.BOX
    #     )

    #scale = image_size / min(*pil_image.size)
    """For 2d medical image center_crop
    """
    if dim == 2:
        a = pil_image.shape
        scale = image_size / min(*a)
        output_size = tuple(round(x * scale) for x in pil_image.shape)
        pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='bicubic').squeeze(0).squeeze(0)
        

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    """For 3d medical image center_crop
    """

    if dim == 3:
        a = pil_image.shape[1:]
        scale = image_size / min(*a)
        output_size = tuple([pil_image.shape[0]]) + tuple(round(x * scale) for x in pil_image.shape[1:])
        pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='trilinear').squeeze(0).squeeze(0)
        

        arr = np.array(pil_image)
        crop_y = (arr.shape[1] - image_size) // 2
        crop_x = (arr.shape[2] - image_size) // 2

        return torch.Tensor(arr[:, crop_y : crop_y + image_size, crop_x : crop_x + image_size])

def center_resize_arr(pil_image, image_size=256):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    #while min(*pil_image.size) >= 2 * image_size:
    # while min(*pil_image.shape) >= 2 * image_size:
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.shape), resample=Image.BOX
    #     )

    #scale = image_size / min(*pil_image.size)
    """For 2d medical image center_crop
    """
    # a = pil_image.shape
    # scale = image_size / min(*a)
    # output_size = tuple(round(x * scale) for x in pil_image.shape)
    # pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='bicubic').squeeze(0).squeeze(0)
    

    # arr = np.array(pil_image)
    # crop_y = (arr.shape[0] - image_size) // 2
    # crop_x = (arr.shape[1] - image_size) // 2
    # return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    """For 3d medical image center_crop
    """
    a = pil_image.shape
    scale = image_size / min(*a)
    output_size = tuple(round(x * scale) for x in pil_image.shape)
    pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='trilinear').squeeze(0).squeeze(0)
    
    #arr = np.array(pil_image)
    crop_z = (image_size - pil_image.shape[0]) // 2
    crop_y = (image_size - pil_image.shape[1]) // 2
    crop_x = (image_size - pil_image.shape[2]) // 2

    # pil_image = np.pad(
    #     arr, pad_width=((crop_z, crop_z), (crop_y,crop_y), (crop_x, crop_x)), mode="constant",  #(256,256,256)
    #     constant_values=0)

    pil_image = F.pad(
            pil_image, [crop_z, crop_z, crop_y, crop_y, crop_x, crop_x], mode="constant",
            value=0
        )

    return pil_image

def center_crop_arr_new(pil_image, image_size=256): #(311,256)

    """For 3d medical image center_crop
    """
    pil_image = np.pad(
        pil_image, pad_width=((0, 0), (25, 26)), mode="constant",  #(311,311)
        constant_values=0)
    # a = pil_image.shape
    # scale = image_size / min(*a)
    # output_size = tuple(round(x * scale) for x in pil_image.shape) #(3,612,512)
    # pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='bicubic').squeeze(0).squeeze(0)  #(512,512)

    a = pil_image.shape[1:]
    scale = image_size / min(*a)
    output_size = tuple([pil_image.shape[0]]) + tuple(round(x * scale) for x in pil_image.shape[1:])
    pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='bicubic').squeeze(0).squeeze(0)  #(6,256,256)
    return pil_image

def center_crop_arr_2d(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    #while min(*pil_image.size) >= 2 * image_size:
    # while min(*pil_image.shape) >= 2 * image_size:
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.shape), resample=Image.BOX
    #     )

    #scale = image_size / min(*pil_image.size)
    """For 2d medical image center_crop
    """
    a = pil_image.shape
    scale = image_size / min(*a)
    output_size = tuple(round(x * scale) for x in pil_image.shape)
    pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='bicubic').squeeze(0).squeeze(0)
    

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    """For 3d medical image center_crop
    """
    # a = pil_image.shape[1:]
    # scale = image_size / min(*a)
    # output_size = tuple([pil_image.shape[0]]) + tuple(round(x * scale) for x in pil_image.shape[1:])
    # pil_image = F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),output_size, mode='trilinear').squeeze(0).squeeze(0)
    

    # arr = np.array(pil_image)
    # crop_y = (arr.shape[1] - image_size) // 2
    # crop_x = (arr.shape[2] - image_size) // 2


    # return arr[:, crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/utils/color_util.py#L186
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img

def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode,[1, 3, 502, 502]
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def to_pil_image(inputs, mem_order, val_range, channel_order):
    # convert inputs to numpy array
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
    assert isinstance(inputs, np.ndarray)
    
    # make sure that inputs is a 4-dimension array
    if mem_order in ["hwc", "chw"]:
        inputs = inputs[None, ...]
        mem_order = f"n{mem_order}"
    # to NHWC
    if mem_order == "nchw":
        inputs = inputs.transpose(0, 2, 3, 1)
    # to RGB
    if channel_order == "bgr":
        inputs = inputs[..., ::-1].copy()
    else:
        assert channel_order == "rgb"
    
    if val_range == "0,1":
        inputs = inputs * 255
    elif val_range == "-1,1":
        inputs = (inputs + 1) * 127.5
    else:
        assert val_range == "0,255"
    
    inputs = inputs.clip(0, 255).astype(np.uint8)
    return [inputs[i] for i in range(len(inputs))]


def put_text(pil_img_arr, text):
    cv_img = pil_img_arr[..., ::-1].copy()
    cv2.putText(cv_img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return cv_img[..., ::-1].copy()


def auto_resize(img: Image.Image, size: int) -> Image.Image:
    short_edge = min(img.size)
    if short_edge < size:
        r = size / short_edge
        img = img.resize(
            tuple(math.ceil(x * r) for x in img.size), Image.BICUBIC
        )
    else:
        # make a deep copy of this image for safety
        img = img.copy()
    return img


def pad(img: np.ndarray, scale: int) -> np.ndarray:
    #h, w = img.shape[:2]
    h, w = img.shape[1:]
    # ph = 0 if h % scale == 0 else math.ceil(h / scale) * scale - h
    # pw = 0 if w % scale == 0 else math.ceil(w / scale) * scale - w
    ph = 0 if scale % h == 0 else math.ceil(scale / h) * scale - h
    pw = 0 if scale % w == 0 else math.ceil(scale / h) * scale - w

    return np.pad(
        img, pad_width=((0, 0), (0, ph), (0, pw)), mode="constant",
        constant_values=0
    )


def center_pad_arr(img, dim=2,image_size=320):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    #while min(*pil_image.size) >= 2 * image_size:
    # while min(*pil_image.shape) >= 2 * image_size:
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.shape), resample=Image.BOX
    #     )

    #scale = image_size / min(*pil_image.size)
    if dim == 2:
        """For 2d medical image center_crop
        """
        h, w = img.shape

        ph = image_size - h
        pw = image_size - w
        ph1 = math.ceil(ph / 2)
        ph2 = ph - ph1
        pw1 = math.ceil(pw / 2)
        pw2 = pw - pw1

        img = np.pad(
            img, pad_width=((ph1, ph2), (pw1, pw2)), mode="constant",
            constant_values=0
        )
        pil_image = torch.Tensor(img.astype("float32")) #F.interpolate(torch.Tensor(img.astype("float32")).unsqueeze(0).unsqueeze(0),(256, 256), mode='bicubic').squeeze(0).squeeze(0)  #(6,256,256)
        return pil_image 
    elif dim == 3:
        """For 3d medical image center_pad
        """
        z, h, w = img.shape
        pz = image_size - z
        ph = image_size - h
        pw = image_size - w
        pz1 = math.ceil(pz / 2)
        pz2 = pz - pz1
        ph1 = math.ceil(ph / 2)
        ph2 = ph - ph1
        pw1 = math.ceil(pw / 2)
        pw2 = pw - pw1

        img = np.pad(
            img, pad_width=((pz1, pz2), (ph1, ph2), (pw1, pw2)), mode="constant",
            constant_values=0
        )
        pil_image = torch.Tensor(img.astype("float32")) #F.interpolate(torch.Tensor(pil_image.astype("float32")).unsqueeze(0).unsqueeze(0),(256, 256), mode='bicubic').squeeze(0).squeeze(0)  #(6,256,256)
        return pil_image 
    elif dim == 4:
        h, w = img.shape[2:4]
        ph = image_size - h
        pw = image_size - w
        ph1 = math.ceil(ph / 2)
        ph2 = ph - ph1
        pw1 = math.ceil(pw / 2)
        pw2 = pw - pw1
        pil_image = F.pad(
            img, [ph1, ph2, pw1, pw2], mode="constant",
            value=0
        )
        #pil_image = torch.Tensor(pil_image.astype("float32"))
        return pil_image 