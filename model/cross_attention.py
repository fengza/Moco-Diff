"""TransFusion from TransFusion: Multi-view Divergent Fusion for Medical Image Segmentation with Transformers."""
from typing import Sequence, Union

import math, copy

import numpy as np
import torch
import torch.nn as nn

# from scripts.view_ops import permute_inverse
# from scripts.view_ops import get_permute_transform


class Attention(nn.Module):

    def __init__(self, num_heads=8, hidden_size=768, atte_dropout_rate=0.0):
        super(Attention, self).__init__()
        # self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(atte_dropout_rate)
        self.proj_dropout = nn.Dropout(atte_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x_1, x_2):    ##（16, 8, 768）
        mixed_query_layer_1 = self.query(x_1)   ##（16, 8, 768）
        # mixed_key_layer_1 = self.key(x_1)       ##（16, 8, 768）
        # mixed_value_layer_1 = self.value(x_1)   ##（16, 8, 768）
        query_layer_1 = self.transpose_for_scores(mixed_query_layer_1) #[16, 24, 8, 32] #[4, 3, 4096, 60]
        # key_layer_1 = self.transpose_for_scores(mixed_key_layer_1)
        # value_layer_1 = self.transpose_for_scores(mixed_value_layer_1)
        #mixed_query_layer_2 = self.query(x_2)
        mixed_key_layer_2 = self.key(x_2)
        mixed_value_layer_2 = self.value(x_2)
        #query_layer_2 = self.transpose_for_scores(mixed_query_layer_2)
        key_layer_2 = self.transpose_for_scores(mixed_key_layer_2)
        value_layer_2 = self.transpose_for_scores(mixed_value_layer_2)

        attention_scores_1 = torch.matmul(query_layer_1,
                                          key_layer_2.transpose(-1, -2))   #[16, 24, 8, 8]  $[4, 3, 4096, 4096]
        attention_scores_1 = attention_scores_1 / math.sqrt(
            self.attention_head_size)
        attention_probs_1 = self.softmax(attention_scores_1)
        # weights_st = attention_probs_st if self.vis else None
        attention_probs_1 = self.attn_dropout(attention_probs_1)
        context_layer_1 = torch.matmul(attention_probs_1, value_layer_2)   #[16, 24, 8, 32]  [4, 3, 4096, 60]
        context_layer_1 = context_layer_1.permute(0, 2, 1, 3).contiguous()  #[16, 8, 24, 32]
        new_context_layer_shape_1 = context_layer_1.size()[:-2] + (          #[16, 8, 768]
            self.all_head_size,)
        context_layer_1 = context_layer_1.view(*new_context_layer_shape_1)
        attention_output_1 = self.out(context_layer_1)                      #[4, 4096, 180], 太大了会不会容易过拟合？
        attention_output_1 = self.proj_dropout(attention_output_1)

        # attention_scores_2 = torch.matmul(query_layer_2,
        #                                   key_layer_1.transpose(-1, -2))
        # attention_scores_2 = attention_scores_2 / math.sqrt(
        #     self.attention_head_size)
        # attention_probs_2 = self.softmax(attention_scores_2)
        # # weights_st = attention_probs_st if self.vis else None
        # attention_probs_2 = self.attn_dropout(attention_probs_2)
        # context_layer_2 = torch.matmul(attention_probs_2, value_layer_1)
        # context_layer_2 = context_layer_2.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape_2 = context_layer_2.size()[:-2] + (
        #     self.all_head_size,)
        # context_layer_2 = context_layer_2.view(*new_context_layer_shape_2)
        # attention_output_2 = self.out(context_layer_2)
        # attention_output_2 = self.proj_dropout(attention_output_2)

        return attention_output_1#, attention_output_2


class Block(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 #mlp_dim=1536,
                 dropout_rate=0.5,
                 num_heads=8,
                 atte_dropout_rate=0.0):
        super(Block, self).__init__()

        #del mlp_dim
        del dropout_rate

        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(num_heads=num_heads,
                              hidden_size=hidden_size,
                              atte_dropout_rate=atte_dropout_rate)

    def forward(self, x_1, x_2):    #（16, 8, 768）
        x_1 = self.attention_norm(x_1)
        x_2 = self.attention_norm(x_2)
        #x_1, x_2 = self.attn(x_1, x_2)
        x_1 = self.attn(x_1, x_2)
        #return x_1, x_2
        return x_1     #[4, 4096, 180]


class TransFusion(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 num_layers: int = 1,
                 #: int = 1536,
                 dropout_rate: float = 0.5,
                 num_heads: int = 8,
                 atte_dropout_rate: float = 0.0,
                 roi_size: Union[Sequence[int], int] = (64, 64),
                 scale: int = 1,
                 #cross_attention_in_origin_view: bool = False
                 ):
        super().__init__()
        if isinstance(roi_size, int):
            roi_size = [roi_size for _ in range(2)]
        #self.cross_attention_in_origin_view = cross_attention_in_origin_view
        patch_size = (1, 1)
        n_patches = (roi_size[0] // patch_size[0] //
                     scale) * (roi_size[1] // patch_size[1] //
                               scale)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        # self.patch_embeddings = nn.Conv2d(in_channels=hidden_size,
        #                                   out_channels=hidden_size,
        #                                   kernel_size=patch_size,
        #                                   stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        for _ in range(num_layers):
            layer = Block(hidden_size=hidden_size,
                          #mlp_dim=mlp_dim,
                          dropout_rate=dropout_rate,
                          num_heads=num_heads,
                          atte_dropout_rate=atte_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x_1, x_2):  #,view_list
        # if self.cross_attention_in_origin_view:
        #     x_1, x_2 = permute_inverse([x_1, x_2], view_list)
        # else:
        #     # Align x_2 to x_1.
        #     x_2 = get_permute_transform(*view_list[::-1])(x_2)
        # x_1 = self.patch_embeddings(x_1)
        # x_2 = self.patch_embeddings(x_2)
        # x_1 = x_1.transpose(-1, -2)    #（16, 8, 768）,[4, 4096, 180]
        # x_2 = x_2.transpose(-1, -2)
        for i in range(x_1.shape[0]):
            
            x_1[i] = x_1[i] + self.position_embeddings      #（16, 8, 768）
            x_2[i] = x_2[i] + self.position_embeddings
        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)   #[4, 4096, 180]
        for layer_block in self.layer:
            #x_1, x_2 = layer_block(x_1, x_2)
            x_1= layer_block(x_1, x_2)    
        x_1 = self.encoder_norm(x_1)
        #x_2 = self.encoder_norm(x_2)
        # B, n_patch, hidden = x_1.size(
        # )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(
        #     np.sqrt(n_patch))
        # x_1 = x_1.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        #x_2 = x_2.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        # if self.cross_attention_in_origin_view:
        #     x_1, x_2 = permute_inverse([x_1, x_2], view_list)
        # else:
        #     x_2 = get_permute_transform(*view_list)(x_2)

        return x_1#, x_2   #[16, 768, 2, 2, 2] [4, 180, 64, 64]
