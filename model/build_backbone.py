import torch
import torch.nn as nn
from .resnet import model_dict


class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=1024):  #head:mlp, feat_dim:128
        super(RGBSingleHead, self).__init__()

        name, width = self._parse_width(name) #resnet50, 1
        dim_in = int(2048 * width)
        self.width = width

        self.encoder = model_dict[name](width=width)

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x, mode=0):  #mode=0
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x = self.encoder(x)
        # feat = x.clone()
        # #if mode == 0 or mode == 1:
        # x = self.head_clip(feat)
        return x

if __name__ == '__main__':
    model = RGBSingleHead()



