# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

# Originally borrowed from DifFace (https://github.com/zsyOAOA/DifFace/blob/master/models/swinir.py)
import sys
import math
from typing import Any, Dict, Set, Mapping, overload
#sys.path.append("/home_data/home/lifeng2023/code/moco/DiffBIR-main/")

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
sys.path.append("/public_bme/data/lifeng/code/moco/TS_BHIR")
from utils.metrics import calculate_psnr_pt, LPIPS
#from .mixins import ImageLoggerMixin
from model.mixins import ImageLoggerMixin
from model.cross_attention import TransFusion
from model.tem_attention import T_BMIR, Fusion_ST
import numpy as np
from PIL import Image
import imageio
from pytorch_lightning.utilities import rank_zero_only
from utils.image import center_pad_arr
from basicsr.archs.arch_util import flow_warp
from model.quality_model import CLIP
from ldm.util import instantiate_from_config

MASK_VIS = []

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH #[225, 6]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  ##[0,1,2,3,4,5,6,7]
        coords_w = torch.arange(self.window_size[1])
        """torch.meshgrid（）的功能是生成网格，可以用于生成坐标。 函数输入两个数据类型相同的一维张量，
        两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数
        """
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww  #分别对应拉直后的像素横坐标和像素纵坐标, (2,64)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):  #[64, 64, 180]
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #[3, 64, 6, 64, 30]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) #[64, 6, 64, 30]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #[64, 6, 64, 64]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)   
        return x   #[64, 64, 180], 64个window,每个window有64个TOKEN,每个TOKEN的向量为180

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttention_mixer(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., is_deformable=True, ori_height=64, ori_width=64):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = 1 #num_heads,原本等于6
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.is_deformable = is_deformable
        self.H = ori_height
        self.W = ori_width

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = True)
        self.project_q = nn.Linear(dim, dim, bias = True)
        self.project_k = nn.Linear(dim, dim, bias = True)

        self.proj_drop = nn.Dropout(proj_drop)
        self.num_windows = None

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.GELU()
        self.ratio = 0.5
        self.route = PredictorLG(dim,window_size[0],ratio=self.ratio) #window_size -> window_size[0]
        # Conv
        k = 3
        d = 2
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d))  
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = True)

    def forward(self, x, mask=None, condition_global=None, train_mode=True): #[128, 64, 180]
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_ori, N_ori, C_ori = x.shape     #B_ => B_ori, N => N_ori, C => C_ori
        x = x.view(-1, x.shape[2], self.H, self.W)  #[2, 180, 64, 64]
        N,C,H,W = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #[3,num_windows*B,head,N,C]
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple), [num_windows*B,head,N,C]
        v = self.project_v(x)            #[2, 180, 64, 64]

        if self.is_deformable:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size[0]),torch.linspace(-1,1,self.window_size[0])))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, self.H//self.window_size[0], self.W//self.window_size[0])  #[2, 2, 64, 64]
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)       #[2, 184, 64, 64]

        mask_mixer, offsets, ca, sa = self.route(_condition,ratio=self.ratio,train_mode=train_mode) #[2, 64, 1],[2, 2, 64, 64],[2, 180, 1, 1],[2, 1, 64, 64]

        q = x    #[2, 180, 64, 64]
        k = x + flow_warp(x, offsets.permute(0,2,3,1), interp_mode='bilinear', padding_mode='border')   #[2, 180, 64, 64]
        qk = torch.cat([q,k],dim=1)

        vs = v*sa    #[2, 180, 64, 64]

        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size[0], dw=self.window_size[0]) #[2, 64, 11520]
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size[0], dw=self.window_size[0]) #[2, 64, 11520]
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size[0], dw=self.window_size[0]) #[2, 64, 23040]

        if train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask_mixer, vs*(1-mask_mixer)   #[2, 64, 11520]
            qk1 = qk*mask_mixer  #[2, 64, 23040]
        else:
            idx1, idx2 = mask_mixer
            _, N_ = idx1.shape
            self.num_windows = N_
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)   ##[1, 15, 11520]
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size[0], dw=self.window_size[0])  #[128, 64, 180]
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size[0], dw=self.window_size[0])  #[2, 4096, 360]

        q1,k1 = torch.chunk(qk1,2,dim=2)  #[2, 4096, 180]
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)           #[2, 4096, 180]
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size[0], dw=self.window_size[0])    #[128, 64, 180]
        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size[0], dw=self.window_size[0])

        #q = q1 * self.scale
        attn = (q1 @ k1.transpose(-2, -1))  # Wh*Ww, Wh*Ww, nH      #[num_windows*B,head,N,N], [128, 64, 64]

        #----before here to revise to token mixer-----

        #f_attn = x.view(-1, (H//self.window_size)*(W//self.window_size), self.window_size * self.window_size * C)   #b n (dh dw c)
        # f_attn = rearrange(x,'(b n) (dh dw) c -> b n (dh dw c)', 
        #     b=N, n=N_, dh=self.window_size, dw=self.window_size)        

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww,[6, 64, 64]
        attn = attn.unsqueeze(1) + relative_position_bias.unsqueeze(0)  #[128, 1, 64, 64]
        attn = attn.squeeze()

        if mask is not None:   #[64,64,64]
            if train_mode:
                nW = mask.shape[0]    #64
                a = mask.unsqueeze(1).unsqueeze(0)  #[1, 64, 1, 64, 64]
                attn = attn.view(B_ori//nW, nW, self.num_heads, N_ori, N_ori) + a  #[2, 64, 1, 64, 64]
                attn = attn.view(-1, self.num_heads, N_ori, N_ori)  #[128, 1, 64, 64]
                attn = self.softmax(attn).squeeze()
            else:
                # mask = batch_index_select(mask,idx1).permute(1, 0, 2).contiguous()   #[15,64,64]
                # attn = attn + mask
                # attn = self.softmax(attn).squeeze()
                nW = mask.shape[0]    #64
                mask = rearrange(mask,'b dh dw -> b (dh dw)', dh=nW, dw=nW).unsqueeze(0)    #[1, 64, 64*64]
                mask = torch.cat([mask]*(B_ori//nW),dim=0)  #[2,64,64*64]
                mask = batch_index_select(mask,idx1)        #[2,15,64*64]
                mask = rearrange(mask,'b n (dh dw) -> (b n) dh dw', dh=nW, dw=nW)    #[2, 15, 64, 64]
                attn = attn + mask  #[30, 64, 64]
                attn = self.softmax(attn).squeeze()
        else:
            attn = self.softmax(attn)

        #attn = self.attn_drop(attn)

        #x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  #这里v改成v1, 后面再接对v2的处理
        f_attn = attn@v1 #[128, 64, 180]

        f_attn = rearrange(f_attn,'(b n) (dh dw) c -> b n (dh dw c)',      #[2, 64, 11520]
            b=N, n=N_, dh=self.window_size[0], dw=self.window_size[0])

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2      #

        attn_out = rearrange(       #B, C, H, W, [2, 180, 64, 64]
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)', 
            h=self.H//self.window_size[0], w=self.W//self.window_size[0], dh=self.window_size[0], dw=self.window_size[0]
        )
        
        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out
        x = self.project_out(out) #[2, 180, 64, 64]
        x= x.view(B_ori, N_ori, C_ori) #[128, 64, 180]

        x = self.proj(x)  #B_, N, C
        x = self.proj_drop(x)
        if not train_mode:
            #mask_mixer = torch.cat(idx1, dim=1)
            return x, idx1
        return x, mask_mixer  #[128, 64, 180]

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        #flops += N * self.dim * 3 * self.dim
        flops +=self.num_windows *  N * self.dim * 1 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_windows * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_windows * self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += self.num_windows * N * self.dim * self.dim
        return flops

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=4,ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k   #180+4=184
        embed_dim = window_size**2
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),    #[2,46,64,64]
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        


    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):    #[2, 184, 64, 64]

        x = self.in_conv(input_x)      #[2, 46, 64, 64]

        offsets = self.out_offsets(x)  #[2, 2, 64, 64]
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)            #[2, 180, 1, 1]
        sa = self.out_SA(x)            #[2, 1, 64, 64]
        
        x = torch.mean(x, keepdim=True, dim=1)      #[2, 1, 64, 64]

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size) #[2, 64, 64]
        B, N, C = x.size()   #[2, 64, 64]

        pred_score = self.out_mask(x)     #[2, 64, 2]
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]     #[2, 64, 1]

        if self.training or train_mode:
            return mask, offsets, ca, sa
        else:
            score = pred_score[:, : , 0]
            B, N = score.shape
            mask_tpk = torch.zeros(B, N, device=score.device)   #[1, 64]
            r = torch.mean(mask,dim=(0,1))*1.0
            if self.ratio == 1:
                num_keep_node = N #int(N * r) #int(N * r)
            else:
                #num_keep_node = min(int(N * r * 2 * self.ratio), N)     #look here!!!!
                num_keep_node = min(int(N * r), N)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]

            #---for mask visualization---
            # out = mask_tpk.reshape(B*N)#.index_fill_(0,idx1,1)
            # out[idx1[0,:]] = 1
            # global MASK_VIS

            # #MASK_VIS += out
            # MASK_VIS.append(out.unsqueeze(0))

            return [idx1, idx2], offsets, ca, sa

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        # self.attn = WindowAttention(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn_mixer = WindowAttention_mixer(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ori_height=self.input_resolution[0], ori_width=self.input_resolution[1])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)  #[64, 64, 64]

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    # def forward(self, x, x_size):   #[1, 4096, 180], (64,64)
    #     H, W = x_size
    #     B, L, C = x.shape
    #     # assert L == H * W, "input feature has wrong size"

    #     shortcut = x
    #     x = self.norm1(x)
    #     x = x.view(B, H, W, C)   #[1, 64, 64, 180]

    #     # cyclic shift
    #     if self.shift_size > 0:
    #         shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    #     else:
    #         shifted_x = x

    #     # partition windows
    #     x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    #     x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    #     # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
    #     if self.input_resolution == x_size:
    #         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    #     else:
    #         attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

    #     # merge windows
    #     attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) #[64, 8, 8, 180]
    #     shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C  #[1, 64, 64, 180]

    #     # reverse cyclic shift
    #     if self.shift_size > 0:
    #         x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    #     else:
    #         x = shifted_x
    #     x = x.view(B, H * W, C)  #[1, 4096, 180]

    #     # FFN
    #     x = shortcut + self.drop_path(x)
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))  #[1, 4096, 180]

    #     return x

    #---revised_to_mixer_tokens------------------
    def forward(self, x, x_size, condition_global=None, train_mode=True):
        H, W = x_size
        B, L, C = x.shape #[2, 4096, 180]
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        #--------from here, we revise the W-MSA and SW-MSA to the mixer tokens------------------
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C, [128, 64, 180]

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_mixer = self.attn_mixer(x_windows, mask=self.attn_mask, condition_global=condition_global, train_mode=train_mode)  # nW*B, window_size*window_size, C, #[128, 64, 180]
            #attn_windows = self.attn_mixer(x_windows, mask=self.attn_mask, condition_global=condition_global)  # nW*B, window_size*window_size, C, #[128, 64, 180]
        else:
            attn_windows, mask_mixer = self.attn_mixer(x_windows, mask=self.calculate_mask(x_size).to(x.device), condition_global=condition_global, train_mode=train_mode)  #[128, 64, 180]
            #attn_windows = self.attn_mixer(x_windows, mask=self.calculate_mask(x_size).to(x.device), condition_global=condition_global)  #[128, 64, 180]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C, [2, 64, 64, 180]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        #---------end------------------ 
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, mask_mixer #[2, 4096, 180]

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    # def forward(self, x, x_size):
    #     for blk in self.blocks:
    #         if self.use_checkpoint:
    #             x = checkpoint.checkpoint(blk, x, x_size)  #[1, 4096, 180]
    #         else:
    #             x = blk(x, x_size)
    #     if self.downsample is not None:
    #         x = self.downsample(x)
    #     return x  #[1, 4096, 180]
    
    #---revised_to_mixer_tokens------------------
    def forward(self, x, x_size, condition_global=None, train_mode=True):
        decision = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x, mask = blk(x, x_size, condition_global=condition_global, train_mode=train_mode)
                decision.append(mask)
                #x = blk(x, x_size, condition_global=condition_global)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, decision

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    # def forward(self, x, x_size):
    #     return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x  #[1, 4096, 180]
    
        # #---revised_to_mixer_tokens------------------
    def forward(self, x, x_size, condition_global=None, train_mode=True):  #[2, 4096, 180],(64,64),[2, 2, 64, 64]
        #return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, condition_global=condition_global), x_size))) + x
        res, mask = self.residual_group(x, x_size, condition_global=condition_global, train_mode=train_mode)
        return self.patch_embed(self.conv(self.patch_unembed(res, x_size))) + x, mask

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):    #[1, 180, 64, 64]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C [1, 4096, 180]
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):  #[1, 4096, 180]
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x   #[1, 180, 64, 64]

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIR(pl.LightningModule, ImageLoggerMixin):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        sf: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        quality_model_config: Mapping[str, Any],
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        sf=8,  #8, revised -> 4
        img_range=1.,
        upsampler='',
        resi_connection='1conv',
        unshuffle=False,
        unshuffle_scale=None,
        hq_key: str="jpg",
        lq_key: str="hint",
        learning_rate: float=None,
        weight_decay: float=None
    ) -> "SwinIR":
        super(SwinIR, self).__init__()
        num_in_ch = in_chans * (unshuffle_scale**2) if unshuffle else in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_size = img_size
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = sf   #8
        self.upsampler = upsampler
        self.window_size = window_size
        self.unshuffle_scale = unshuffle_scale  #8
        self.unshuffle = unshuffle
        # self.temporal_block = T_BMIR()
        # self.fusion_block = Fusion_ST()
        
        #####################################################################################################
        #################################### 0, get image quality score #####################################
        self.quality_predictor: CLIP = instantiate_from_config(quality_model_config)

        #gate_1的概率
        self.gate_score_1 = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        ################################### 1, shallow feature extraction ###################################
        if unshuffle:
            # assert unshuffle_scale is not None
            # self.conv_first_revised = nn.Sequential(
            #     nn.PixelUnshuffle(sf),                     #img缩小sf倍，channel=channel*sf**2倍
            #     nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1),  #不改变img大小，只改变channel
            # )
            self.conv_first = nn.Sequential(
                nn.PixelUnshuffle(sf),                     #img缩小sf倍，channel=channel*sf**2倍
                nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1),  #不改变img大小，只改变channel
            )
        else:
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.basic_num_blocks = np.sum(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        #self.x_after_conv_first = nn.Parameter(torch.zeros((12, 6, self.embed_dim, 64, 64))) #[B, 6, 180, 64, 64]]


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.global_predictor = nn.Sequential(nn.Conv2d(embed_dim, 8, 1, 1, 0, bias=True),   #把n_feat换成embed_dim了，不懂n_feat代表channels
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(8, 2, 3, 1, 1, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):   #8个RSTB
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(   #这里有问题
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )
        

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(sf, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                sf, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1])
            )
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            elif self.upscale == 8:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
        
        self.hq_key = hq_key
        self.lq_key = lq_key
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lpips_metric = LPIPS(net="alex")
        # self.cross_atte6 = TransFusion(
        #     hidden_size=embed_dim,
        #     num_layers=1,
        #     #mlp_dim=feature_size * 32,
        #     num_heads=3,
        #     dropout_rate=0.0,
        #     atte_dropout_rate=0.0,
        #     roi_size=(patches_resolution[0], patches_resolution[1]),
        #     scale=1)
        

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # TODO: What's this ?
    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> Set[str]:
        return {'relative_position_bias_table'}

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def rout_path(self, x: torch.Tensor, train_mode=True) -> torch.Tensor: 
        pred_score = self.gate_score_1(x)     #[2, 2, 2]
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]     #[2, 8, 2]
        #return mask
        if train_mode:
            return mask
        else:
            score = pred_score[:, : , 0]
            return score

    #def forward_features(self, x: torch.Tensor, tempotal_x: torch.Tensor, train_mode=True) -> torch.Tensor:
    def forward_features(self, x: torch.Tensor, train_mode=True) -> torch.Tensor:
    #def forward_features(self, x: torch.Tensor, tempotal_x: torch.Tensor) -> torch.Tensor:  #[1, 180, 64, 64], -> #[B,6,180,64,64] , tempotal_x:[B,180,6,64,64]
    #def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        #x, x_plus, x_minus = x_combine
        decision = []
        x_size = (x.shape[2], x.shape[3])
        condition_global = self.global_predictor(x)       #[2, 2, 64, 64]

        x = self.patch_embed(x)  ##[1, 4096, 180]
        # x_plus = self.patch_embed(x_plus)
        # x_minus = self.patch_embed(x_minus)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        #record_i = 0
        # heatmap = torch.zeros_like(x)
        for layer in self.layers:     #开始8个RSTB模块,开始结束x大小不变，[1, 4096, 180]
            """在第一个layer输出的时候，联合t输入，进入Fusion模块，得到的输出再进入layer 2(都是单帧)
            """
            #record_i += 1
            #x = layer(x, x_size)      #在这里，给2个input x1 x2加入self-attention, 对其至supporting slice(x1)
            x, mask = layer(x, x_size, condition_global=condition_global,train_mode=train_mode)
            decision.extend(mask)  #[b,6,被选中的idx数量] -> [b,6*8,被选中的idx数量]
            # if record_i == 1:
            #     B,C,D,H,W = tempotal_x.shape  #[B,180,6,64,64]
            #     x = rearrange(x.view(B,H,W,-1), "n h w c -> n c h w")  #[B, 180, 64, 64]
            #     x = self.fusion_block(x, tempotal_x[:,:,2,...])        #[B, 180, 64, 64]
            # # #x = rearrange(x,"n c h w -> n h w c").contiguous()
            #     x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)      #[B, 4096, 180]
            # if record_i == 8:
            #     heatmap = x
            # x_plus = layer(x_plus, x_size)
            # x_minus = layer(x_minus, x_size)

            # x = self.cross_atte6(x, x_plus)
            # x = self.cross_atte6(x, x_minus)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        decision = torch.cat(decision,dim=1)
        #heatmap = self.patch_unembed(heatmap, x_size)

        #return x#, heatmap    #[B, 180, 64, 64]
        return x, decision
    
    def dbt_plus(self, x: torch.Tensor, train_mode=True):
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # for i in range(num_slices):
        #     x_now = x[:,i,...]

        #     self.mean = self.mean.type_as(x_now)
        #     x_now = (x_now - self.mean) * self.img_range
        #     x[:,i,...] = x_now

        # self.mean_plus = self.mean.type_as(x_plus)
        # x_plus = (x_plus - self.mean_plus) * self.img_range

        # self.mean_minus = self.mean.type_as(x_minus)
        # x_minus = (x_minus - self.mean_minus) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            #x = self.conv_first_revised(x)    #[1, 180, 64, 64], [1,]

            x = self.conv_first(x)
            shortcut = x
            x, decision = self.forward_features(x,train_mode=train_mode)
            x = self.conv_after_body(x) + shortcut  #[1, 180, 64, 64]
            
            #B, D, C, H, W = x.shape    # [b,6,3,512,512]
            # #x_after_conv_first = nn.Parameter(torch.zeros((B, D, self.embed_dim, H // self.upscale, W // self.upscale))) #[B, 6, 180, 64, 64]]
            # """把这个地方重新改下，以前这种写法必须要fix batch size
            # """
            # #-----start-----#
            # for i in range(num_slices):
            #     #x_now = x[:,i,...]
            #     #x_now = self.conv_first(x_now)  #[B, 180, 64, 64]
            #     #self.x_after_conv_first[:,i,...] = self.conv_first(x_now) #x_now
            #     with torch.no_grad():
            #         self.x_after_conv_first[:,i,...] = self.conv_first(x[:,i,...])

            # x = self.x_after_conv_first
            # x.requires_grad_(True)
            #-----end-----#
            
            """重写如下
            """
            #-----start-----#
            # x = x.view(-1, C, H, W)    # [b,6,3,512,512]->[6b,3,512,512]
            # x = self.conv_first(x).view(B, D, self.embed_dim, int(H/self.upscale), int(W/self.upscale))   #[6b,180,64,64] -> [b,6,180,64,64]  
            # #-----end-----#

            #x_key = x[:,2,...]  #[b,180,64,64]
            
            #x_key = x[:,2,...].clone()
            # x_plus = self.conv_first(x_plus)
            # x_minus = self.conv_first(x_minus)
            #tempotal_x = self.temporal_block(x)  #[b, 180, 6, 64, 64]
            #x,heatmap = self.forward_features(x_key,tempotal_x)  ##[12, 180, 64, 64]
            #x = self.forward_features(x_key,tempotal_x)    #[2, 180, 64, 64]
            #x, decision = self.forward_features(x_key,tempotal_x,train_mode=train_mode)    #[2, 180, 64, 64]
            #heatmap_ori = heatmap
            
            #x = self.conv_after_body(x) + x_key  #[1, 180, 64, 64]
        
            x = self.conv_before_upsample(x)   #[1, 64, 64, 64]
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))  #[1, 64, 128, 128]

            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            elif self.upscale == 8:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
                x = self.lrelu(self.conv_up3(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest'))) #[1, 64, 512, 512] #不经过这个

                # heatmap = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(heatmap, scale_factor=2, mode='nearest')))
                # heatmap = self.lrelu(self.conv_up3(torch.nn.functional.interpolate(heatmap, scale_factor=2, mode='nearest'))) #[1, 64, 512, 512]
            x = self.conv_last(self.lrelu(self.conv_hr(x))) #[1, 3, 512, 512]

            #heatmap = self.conv_last(self.lrelu(self.conv_hr(heatmap)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean  #[2, 3, 512, 512]

        return x#, decision 

    def forward(self, x: torch.Tensor, train_mode=True) -> torch.Tensor:  #[1, 3, 512, 512]*3, [B,6,3,512,512]
        
        # num_slices = x.shape[1]
        #H, W = x.shape[3:]
        #quality_score = self.quality_predictor(x[:,2,...]) #[2, 5], 再产出一个[2,2]
        B, C, H, W = x.shape
        quality_score, img_score = self.quality_predictor(x)  #把img_score做成  img_score * R(x) + (1-img_score)*x的样子，直接后接整个网络的mse，分成train和infer两种
        # H, W = x.shape[2:]
        _, preds = torch.max(img_score,1)
        preds = preds.unsqueeze(1)
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        ones = torch.ones(preds.shape).to(device=img_score.device)
        x_no = ((ones-preds) * x.reshape(B,-1)).reshape(B, C, H, W)
        x_no = x_no / self.img_range + self.mean 

        x_op = self.dbt_plus((preds * x.reshape(B,-1)).reshape(B, C, H, W), train_mode)

        x = x_no + x_op
        # if train_mode:
        #     no_op_res = (ones-img_score) * x.reshape(B,-1)
        #     no_op_res = no_op_res.reshape(B, C, H, W)
        #     x = self.check_image_size(x)
        #     self.mean = self.mean.type_as(x)
        #     x = (x - self.mean) * self.img_range

        #     # for i in range(num_slices):
        #     #     x_now = x[:,i,...]

        #     #     self.mean = self.mean.type_as(x_now)
        #     #     x_now = (x_now - self.mean) * self.img_range
        #     #     x[:,i,...] = x_now

        #     # self.mean_plus = self.mean.type_as(x_plus)
        #     # x_plus = (x_plus - self.mean_plus) * self.img_range

        #     # self.mean_minus = self.mean.type_as(x_minus)
        #     # x_minus = (x_minus - self.mean_minus) * self.img_range

        #     if self.upsampler == 'pixelshuffle':
        #         # for classical SR
        #         x = self.conv_first(x)
        #         x = self.conv_after_body(self.forward_features(x)) + x
        #         x = self.conv_before_upsample(x)
        #         x = self.conv_last(self.upsample(x))
        #     elif self.upsampler == 'pixelshuffledirect':
        #         # for lightweight SR
        #         x = self.conv_first(x)
        #         x = self.conv_after_body(self.forward_features(x)) + x
        #         x = self.upsample(x)
        #     elif self.upsampler == 'nearest+conv':
        #         # for real-world SR
        #         #x = self.conv_first_revised(x)    #[1, 180, 64, 64], [1,]

        #         x = self.conv_first(x)
        #         shortcut = x
        #         x, decision = self.forward_features(x,train_mode=train_mode)
        #         x = self.conv_after_body(x) + shortcut  #[1, 180, 64, 64]
                
        #         #B, D, C, H, W = x.shape    # [b,6,3,512,512]
        #         # #x_after_conv_first = nn.Parameter(torch.zeros((B, D, self.embed_dim, H // self.upscale, W // self.upscale))) #[B, 6, 180, 64, 64]]
        #         # """把这个地方重新改下，以前这种写法必须要fix batch size
        #         # """
        #         # #-----start-----#
        #         # for i in range(num_slices):
        #         #     #x_now = x[:,i,...]
        #         #     #x_now = self.conv_first(x_now)  #[B, 180, 64, 64]
        #         #     #self.x_after_conv_first[:,i,...] = self.conv_first(x_now) #x_now
        #         #     with torch.no_grad():
        #         #         self.x_after_conv_first[:,i,...] = self.conv_first(x[:,i,...])

        #         # x = self.x_after_conv_first
        #         # x.requires_grad_(True)
        #         #-----end-----#
                
        #         """重写如下
        #         """
        #         #-----start-----#
        #         # x = x.view(-1, C, H, W)    # [b,6,3,512,512]->[6b,3,512,512]
        #         # x = self.conv_first(x).view(B, D, self.embed_dim, int(H/self.upscale), int(W/self.upscale))   #[6b,180,64,64] -> [b,6,180,64,64]  
        #         # #-----end-----#

        #         #x_key = x[:,2,...]  #[b,180,64,64]
                
        #         #x_key = x[:,2,...].clone()
        #         # x_plus = self.conv_first(x_plus)
        #         # x_minus = self.conv_first(x_minus)
        #         #tempotal_x = self.temporal_block(x)  #[b, 180, 6, 64, 64]
        #         #x,heatmap = self.forward_features(x_key,tempotal_x)  ##[12, 180, 64, 64]
        #         #x = self.forward_features(x_key,tempotal_x)    #[2, 180, 64, 64]
        #         #x, decision = self.forward_features(x_key,tempotal_x,train_mode=train_mode)    #[2, 180, 64, 64]
        #         #heatmap_ori = heatmap
                
        #         #x = self.conv_after_body(x) + x_key  #[1, 180, 64, 64]
            
        #         x = self.conv_before_upsample(x)   #[1, 64, 64, 64]
        #         x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))  #[1, 64, 128, 128]

        #         if self.upscale == 4:
        #             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #         elif self.upscale == 8:
        #             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #             x = self.lrelu(self.conv_up3(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest'))) #[1, 64, 512, 512] #不经过这个

        #             # heatmap = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(heatmap, scale_factor=2, mode='nearest')))
        #             # heatmap = self.lrelu(self.conv_up3(torch.nn.functional.interpolate(heatmap, scale_factor=2, mode='nearest'))) #[1, 64, 512, 512]
        #         x = self.conv_last(self.lrelu(self.conv_hr(x))) #[1, 3, 512, 512]

        #         #heatmap = self.conv_last(self.lrelu(self.conv_hr(heatmap)))
        #     else:
        #         # for image denoising and JPEG compression artifact reduction
        #         x_first = self.conv_first(x)
        #         res = self.conv_after_body(self.forward_features(x_first)) + x_first
        #         x = x + self.conv_last(res)

        #     x = x / self.img_range + self.mean  #[2, 3, 512, 512]

        #     op_res = img_score * x.reshape(B,-1)
        #     op_res = op_res.reshape(B,C,H,W)
        #     res = op_res + no_op_res
        #     return res, decision, quality_score, img_score
        #     #return x[:, :, :H*self.upscale, :W*self.upscale]#, heatmap[:, :, :H*self.upscale, :W*self.upscale], heatmap_ori
        #     #return x
        #     # if not train_mode:
        #     #     global MASK_VIS
        #     #     MASK_VIS = torch.cat(MASK_VIS, dim=0)  #[48,64]
        #     #     B, _ = MASK_VIS.shape
        #     #     vis_mask = MASK_VIS.reshape(B,-1,64)
        #     #     MASK_VIS = []
        #     #     return x, decision, quality_score, img_score,vis_mask
        #     # else:
        #     #     return x, decision, quality_score, img_score
        
        # else: #val_batch = 1
        #     if img_score[0]:
        #         x = self.check_image_size(x)
        #         self.mean = self.mean.type_as(x)
        #         x = (x - self.mean) * self.img_range

        #         # for i in range(num_slices):
        #         #     x_now = x[:,i,...]

        #         #     self.mean = self.mean.type_as(x_now)
        #         #     x_now = (x_now - self.mean) * self.img_range
        #         #     x[:,i,...] = x_now

        #         # self.mean_plus = self.mean.type_as(x_plus)
        #         # x_plus = (x_plus - self.mean_plus) * self.img_range

        #         # self.mean_minus = self.mean.type_as(x_minus)
        #         # x_minus = (x_minus - self.mean_minus) * self.img_range

        #         if self.upsampler == 'pixelshuffle':
        #             # for classical SR
        #             x = self.conv_first(x)
        #             x = self.conv_after_body(self.forward_features(x)) + x
        #             x = self.conv_before_upsample(x)
        #             x = self.conv_last(self.upsample(x))
        #         elif self.upsampler == 'pixelshuffledirect':
        #             # for lightweight SR
        #             x = self.conv_first(x)
        #             x = self.conv_after_body(self.forward_features(x)) + x
        #             x = self.upsample(x)
        #         elif self.upsampler == 'nearest+conv':
        #             # for real-world SR
        #             #x = self.conv_first_revised(x)    #[1, 180, 64, 64], [1,]

        #             x = self.conv_first(x)
        #             shortcut = x
        #             x, decision = self.forward_features(x,train_mode=train_mode)
        #             x = self.conv_after_body(x) + shortcut  #[1, 180, 64, 64]
                    
        #             #B, D, C, H, W = x.shape    # [b,6,3,512,512]
        #             # #x_after_conv_first = nn.Parameter(torch.zeros((B, D, self.embed_dim, H // self.upscale, W // self.upscale))) #[B, 6, 180, 64, 64]]
        #             # """把这个地方重新改下，以前这种写法必须要fix batch size
        #             # """
        #             # #-----start-----#
        #             # for i in range(num_slices):
        #             #     #x_now = x[:,i,...]
        #             #     #x_now = self.conv_first(x_now)  #[B, 180, 64, 64]
        #             #     #self.x_after_conv_first[:,i,...] = self.conv_first(x_now) #x_now
        #             #     with torch.no_grad():
        #             #         self.x_after_conv_first[:,i,...] = self.conv_first(x[:,i,...])

        #             # x = self.x_after_conv_first
        #             # x.requires_grad_(True)
        #             #-----end-----#
                    
        #             """重写如下
        #             """
        #             #-----start-----#
        #             # x = x.view(-1, C, H, W)    # [b,6,3,512,512]->[6b,3,512,512]
        #             # x = self.conv_first(x).view(B, D, self.embed_dim, int(H/self.upscale), int(W/self.upscale))   #[6b,180,64,64] -> [b,6,180,64,64]  
        #             # #-----end-----#

        #             #x_key = x[:,2,...]  #[b,180,64,64]
                    
        #             #x_key = x[:,2,...].clone()
        #             # x_plus = self.conv_first(x_plus)
        #             # x_minus = self.conv_first(x_minus)
        #             #tempotal_x = self.temporal_block(x)  #[b, 180, 6, 64, 64]
        #             #x,heatmap = self.forward_features(x_key,tempotal_x)  ##[12, 180, 64, 64]
        #             #x = self.forward_features(x_key,tempotal_x)    #[2, 180, 64, 64]
        #             #x, decision = self.forward_features(x_key,tempotal_x,train_mode=train_mode)    #[2, 180, 64, 64]
        #             #heatmap_ori = heatmap
                    
        #             #x = self.conv_after_body(x) + x_key  #[1, 180, 64, 64]
                
        #             x = self.conv_before_upsample(x)   #[1, 64, 64, 64]
        #             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))  #[1, 64, 128, 128]

        #             if self.upscale == 4:
        #                 x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #             elif self.upscale == 8:
        #                 x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #                 x = self.lrelu(self.conv_up3(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest'))) #[1, 64, 512, 512] #不经过这个

        #                 # heatmap = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(heatmap, scale_factor=2, mode='nearest')))
        #                 # heatmap = self.lrelu(self.conv_up3(torch.nn.functional.interpolate(heatmap, scale_factor=2, mode='nearest'))) #[1, 64, 512, 512]
        #             x = self.conv_last(self.lrelu(self.conv_hr(x))) #[1, 3, 512, 512]

        #             #heatmap = self.conv_last(self.lrelu(self.conv_hr(heatmap)))
        #         else:
        #             # for image denoising and JPEG compression artifact reduction
        #             x_first = self.conv_first(x)
        #             res = self.conv_after_body(self.forward_features(x_first)) + x_first
        #             x = x + self.conv_last(res)

        #         x = x / self.img_range + self.mean  #[2, 3, 512, 512]
        #     else:
        #         x = x
        #         decision = []
            # global MASK_VIS
            # MASK_VIS = torch.cat(MASK_VIS, dim=0)  #[48,64]
            # B, _ = MASK_VIS.shape
            # vis_mask = MASK_VIS.reshape(B,-1,64)
            # MASK_VIS = []
            
        return x, img_score#,vis_mask            

    def flops(self) -> int:
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
    
    def get_loss(self, pred: torch.Tensor, label: torch.Tensor, decision: torch.Tensor, quality_score: torch.Tensor,train_mode=True) -> torch.Tensor:
    #def get_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between model predictions and labels.
        
        Args:
            pred (torch.Tensor): Batch model predictions.
            label (torch.Tensor): Batch labels.
        
        Returns:
            loss (torch.Tensor): The loss tensor.
        """
        #loss = F.mse_loss(input=pred, target=label, reduction="sum")
        loss = torch.abs(pred-label).sum(dim=(0,1,2,3))
        return loss
        #return torch.abs(pred-label).sum(dim=(1,2,3))

        #---following is for token-level routing---

        # l1 = 1.5 * F.mse_loss(input=pred, target=label)  #1.5;0.4;0.5
        # # #mask = torch.cat(decision,dim=1)
        # if train_mode:
        #     #ratio_setting = torch.tensor(np.array([0.05, 0.2, 0.5, 0.8, 0.99]*pred.shape[0])).reshape(pred.shape[0],-1).to(device=pred.device) #[B,5] #.reshape(pred.shape[0], 1)
        #     ratio_setting = torch.tensor(np.array([0.005, 0.4, 0.999]*pred.shape[0])).reshape(pred.shape[0],-1).to(device=pred.device) #[B,5]
        #     #l_ratio = torch.zeros(pred.shape[0], 1).to(device=pred.device) #[B,1]
        #     decision = torch.mean(decision,dim=(1))    #[B,1]
        #     #decision = torch.mean(decision,dim=(0,1))  #这里出错，len=64*48，没有的地方由0代替，有的地方为1，但是infer时，给的是index
        #     decision = decision * quality_score  #[B,5]
        #     ratio_selected = quality_score * ratio_setting  #[B,5]
        #     # idx_nonzero = torch.nonzero(decision)[1]   
        #     # ratio_selected = ratio_setting[idx_nonzero] #[B,1]
        #     # for i in range(decision.shape[0]):
        #     #     l_ratio[i] = decision[i].squeeze()  #[B,1]
        #     l_ratio_mixed = 0.5*torch.sum((decision-ratio_selected)**2, dim=(0,1))
        #     loss = l1 + l_ratio_mixed
        # else:
        #     device = decision.device
        #     B, N = decision.shape
        #     batch = pred.shape[0]
        #     sums = batch * self.img_size * self.basic_num_blocks
        #     decision = torch.tensor((B*N)/sums).to(device=device)
        #     l_ratio = 0.5*(decision-0.7)**2  #self.ratio=0.5
        # # #decision = torch.mean(decision,dim=(0,1))
        #     #loss = l1
        # # #l_ratio = 2*0.5*(decision-0.7)**2  #self.ratio=0.5
        #     loss = l1 + l_ratio
        # return loss
    
    # def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
    #     """
    #     Args:
    #         batch (Dict[str, torch.Tensor]): A dict contains LQ and HQ (NHWC, RGB, 
    #             LQ range in [0, 1] and HQ range in [-1, 1]).
    #         batch_idx (int): Index of this batch.
        
    #     Returns:
    #         outputs (torch.Tensor): The loss tensor.
    #     """
    #     hq, lq = batch[self.hq_key], batch[self.lq_key]
    #     # batch0 = batch[0]
    #     # batch_plus = batch[1]
    #     # batch_minus = batch[2]
    #     # hq, lq = batch0[self.hq_key], batch0[self.lq_key]
    #     # hq_plus, lq_plus = batch_plus[self.hq_key], batch_plus[self.lq_key]
    #     # hq_minus, lq_minus = batch_minus[self.hq_key], batch_minus[self.lq_key]
    #     hq = rearrange(((hq + 1) / 2).clamp_(0, 1), "n h w c -> n c h w")
    #     #lq = rearrange(lq, "n h w c -> n c h w")
    #     lq = rearrange(lq, "n d h w c -> n d c h w")
    #     pred = self(lq)
    #     #pred = center_pad_arr(pred,4,320)
    #     # lq_minus = rearrange(lq_minus, "n h w c -> n c h w")
    #     #pred = self((lq, lq_plus, lq_minus))
    #     loss = self.get_loss(pred, hq)
    #     self.log("train_loss", loss, on_step=True,rank_zero_only=True)
    #     return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Args:
            batch (Dict[str, torch.Tensor]): A dict contains LQ and HQ (NHWC, RGB, 
                LQ range in [0, 1] and HQ range in [-1, 1]).
            batch_idx (int): Index of this batch.
        
        Returns:
            outputs (torch.Tensor): The loss tensor.
        """
        hq, lq = batch[self.hq_key], batch[self.lq_key]
        hq = rearrange(((hq + 1) / 2).clamp_(0, 1), "n h w c -> n c h w")
        #lq = rearrange(lq, "n d h w c -> n d c h w")
        lq = rearrange(lq, "n h w c -> n c h w")
        #--- for visualization---
        #pred, decision, quality_score = self(lq, train_mode=True)
        pred, img_routing_score = self(lq, train_mode=True)
        decision = []
    #     #-------create the gt for classifier-------
        # _, preds = torch.max(img_routing_score, 1)
        a = torch.absolute(lq-hq).sum(dim=(1,2,3))
        b = torch.absolute(pred-hq).sum(dim=(1,2,3))
        discriminate = a > b
        routing_true_label = torch.tensor([1 if i == True else 0 for i in discriminate]).to(device='cuda:0')
        #outing_true = F.one_hot(routing_true_label).float()

    #     #-----get the classification loss-----
        criterion=nn.CrossEntropyLoss(reduction="mean")
        loss_gate=criterion(img_routing_score,routing_true_label)
        #loss_mse = self.get_loss(pred, hq, preds, train_mode=True)
        #loss = self.get_loss(pred, hq, decision, quality_score, train_mode=True)
        self.log("train_loss_gate", loss_gate, on_step=True)
        return loss_gate

    

    #@rank_zero_only
    def on_validation_start(self) -> None:
        self.lpips_metric.to(self.device)
        #pass

    #@torch.no_grad()
    # def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        
    #     #batch0 = batch[0]
    #     # batch_plus = batch[1]
    #     # batch_minus = batch[2]
    #     hq, lq = batch[self.hq_key], batch[self.lq_key]
    #     # hq_plus, lq_plus = batch_plus[self.hq_key], batch_plus[self.lq_key]
    #     # hq_minus, lq_minus = batch_minus[self.hq_key], batch_minus[self.lq_key]
    #     lq = rearrange(lq, "n d h w c -> n d c h w")
    #     #lq = rearrange(lq, "n h w c -> n c h w")
    #     # lq_minus = rearrange(lq_minus, "n h w c -> n c h w")
    #     pred = self(lq)
    #     #pred = center_pad_arr(pred,4,320)
    #     #pred = self((lq, lq_plus, lq_minus))
    #     hq = rearrange(((hq + 1) / 2).clamp_(0, 1), "n h w c -> n c h w")
        
    #     # requiremtns for lpips model inputs:
    #     # https://github.com/richzhang/PerceptualSimilarity
    #     lpips = self.lpips_metric(pred, hq, normalize=True).mean()
    #     self.log("val_lpips", lpips,rank_zero_only=True)
        
    #     pnsr = calculate_psnr_pt(pred, hq, crop_border=0).mean()
    #     self.log("val_pnsr", pnsr,rank_zero_only=True)
        
    #     loss = self.get_loss(pred, hq)
    #     self.log("val_loss", loss,rank_zero_only=True)
    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        hq, lq = batch[self.hq_key], batch[self.lq_key]
        #running_corrects = 0
        #lq = rearrange(lq, "n d h w c -> n d c h w")
        lq = rearrange(lq, "n h w c -> n c h w")
        #--- for visualization---
        #pred, decision, quality_score, vis_mask = self(lq, train_mode=False)
        pred, img_routing_score = self(lq, train_mode=False) #, vis_mask
        hq = rearrange(((hq + 1) / 2).clamp_(0, 1), "n h w c -> n c h w")
        
        a = torch.absolute(lq-hq).sum(dim=(1,2,3))
        b = torch.absolute(pred-hq).sum(dim=(1,2,3))
        discriminate = a > b
        routing_true_label = torch.tensor([1 if i == True else 0 for i in discriminate]).to(device='cuda:0')    #1: op; 0: no_op
        img_routing_score = img_routing_score.squeeze()
        #routing_true = F.one_hot(routing_true_label).float()
        criterion=nn.CrossEntropyLoss(reduction="mean")
        loss_gate=criterion(img_routing_score,routing_true_label)
        # _, preds = torch.max(img_routing_score, 1)
        # routing_true[routing_true == True] = 1
        # routing_true[routing_true == False] = 0
        # # requiremtns for lpips model inputs:
        # # https://github.com/richzhang/PerceptualSimilarity
        lpips = self.lpips_metric(pred, hq, normalize=True).mean()
        self.log("val_lpips", lpips)
        
        pnsr = calculate_psnr_pt(pred, hq, crop_border=0).mean()
        self.log("val_pnsr", pnsr)

        self.log("val_gate_loss", loss_gate)
        
        # loss = self.get_loss(pred, hq, decision, quality_score, train_mode=False)
        # self.log("val_loss", loss)
    
    def configure_optimizers(self) -> optim.AdamW:
        """
        Configure optimizer for this model.
        
        Returns:
            optimizer (optim.AdamW): The optimizer for this model.
        """
        optimizer = optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    # @torch.no_grad()
    # def log_images(self, batch: Any) -> Dict[str, torch.Tensor]:
    #     hq, lq = batch[self.hq_key], batch[self.lq_key]
    #     hq = rearrange(((hq + 1) / 2).clamp_(0, 1), "n h w c -> n c h w")
    #     # rgb_mean = (0.4488, 0.4371, 0.4040)
    #     # mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1).cuda()
    #     lq = rearrange(lq, "n d h w c -> n d c h w")
    #     #lq = rearrange(lq, "n h w c -> n c h w")
    #     pred = self(lq)
    #     #pred = center_pad_arr(pred,4,320)
    #     #lq = lq[:,2,...]

    #     # a = lq[:,2,...] 
    #     # b = a + mean
    #     # grid = lq[:,2,...].transpose(0, 1).transpose(1, 2).squeeze(-1).cpu().detach().numpy()
    #     # #grid = grid.numpy()
    #     # grid = (grid * 255).clip(0, 255).astype(np.uint8)
    #     # new_im_hq = Image.fromarray(grid)  # 调用Image库，数组归一化
    #     # # # 保存图片到本地
    #     # imageio.imsave(f'/public_bme/data/lifeng/data/moco/test/test_hq.jpg', new_im_hq)
    #     #grid = sitk.GetImageFromArray(grid)
    #     # batch0 = batch[0]
    #     # batch_plus = batch[1]
    #     # batch_minus = batch[2]
    #     # hq, lq = batch0[self.hq_key], batch0[self.lq_key]
    #     # hq_plus, lq_plus = batch_plus[self.hq_key], batch_plus[self.lq_key]
    #     # hq_minus, lq_minus = batch_minus[self.hq_key], batch_minus[self.lq_key]
      
    #     # lq_plus = rearrange(lq_plus, "n h w c -> n c h w")
    #     # lq_minus = rearrange(lq_minus, "n h w c -> n c h w")
    #     # #pred = self(lq)
    #     # pred = self((lq, lq_plus, lq_minus))
    #     # hq = rearrange(((hq + 1) / 2).clamp_(0, 1), "n h w c -> n c h w")
    #     return dict(lq=lq[:,2,...], pred=pred, hq=hq)

    @torch.no_grad()
    def log_images(self, batch: Any) -> Dict[str, torch.Tensor]:
        hq, lq = batch[self.hq_key], batch[self.lq_key]
        #hq, lq = batch[self.hq_key][0].unsqueeze(0), batch[self.lq_key][0].unsqueeze(0)
        hq = rearrange(((hq + 1) / 2).clamp_(0, 1), "n h w c -> n c h w")
        lq = rearrange(lq, "n h w c -> n c h w")
        #lq = rearrange(lq, "n d h w c -> n d c h w")
        # pred = self(lq)
        # return dict(lq=lq, pred=pred, hq=hq)
        #--- for visualization---
        pred, img_routing_score = self(lq, train_mode=False) #, vis_mask
        #pred, decision, quality_score, img_routing_score = self(lq, train_mode=False)
        #decision = decision.shape[1]
        # mixer = 0.36 #round(decision / (64*48),3)
        # idx_path = int(torch.nonzero(quality_score)[0][1])
        #return dict(lq=lq, pred=pred, hq=hq), mixer, idx_path#, int(img_routing_score[0])
        #-------create the gt for classifier-------
        # a = F.mse_loss(input=lq, target=hq)#.sum(dim=(1,2,3))
        # b = F.mse_loss(input=pred, target=hq)#.sum(dim=(1,2,3))
        a = torch.abs(lq-hq).sum(dim=(1,2,3))[0]
        b = torch.abs(pred-hq).sum(dim=(1,2,3))[0]
        discriminate = a > b
        #routing_true_label = torch.tensor([1 if i == True else 0 for i in discriminate]).to(device='cuda:0')
        if discriminate:
            routing_true_label = 1
        else:
            routing_true_label = 0
        routing_true_label = torch.tensor(routing_true_label).to(device='cuda:0')
        # #-------create the preds for classifier-------
        #_, preds = torch.max(img_routing_score, 1)
        what, preds = torch.max(img_routing_score,1)
        #preds = img_routing_score
        #return dict(lq=lq, pred=pred, hq=hq), mixer, idx_path, int(preds[0]), int(routing_true_label[0])
        return dict(lq=lq, pred=pred, hq=hq), int(preds[0]), int(routing_true_label)