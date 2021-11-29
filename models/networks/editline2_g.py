import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize
from models.networks.base_network import BaseNetwork
from models.networks.utils import gen_conv, gen_deconv, dis_conv
from models.networks.splitcam import ReduceContextAttentionP1, ReduceContextAttentionP2


class MDGenerator(BaseNetwork):
    def __init__(self, opt):
        super(MDGenerator, self).__init__()
        cnum=48
        # stage1
        self.conv1 = gen_conv(4, cnum, 5, 1)
        self.conv2_downsample = gen_conv(int(cnum/2), 2*cnum, 3, 2)
        self.conv3 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.conv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.conv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.conv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.conv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16)

        self.conv11 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv12 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv14 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum//2, cnum//2, 3, 1)
        self.conv17 = gen_conv(cnum//4, 3, 3, 1, activation=None)
        # stage mask
        self.conv_mask_11 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv_mask_12 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv_mask_13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv_mask_14 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv_mask_15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv_mask_16 = gen_conv(cnum//2, cnum//2, 3, 1)
        self.conv_mask_17 = gen_conv(cnum//4, 1, 3, 1, activation=None)


    def get_param_list(self, stage="all"):
        if stage=="all" or stage=="mask":
            list_param = [p for name, p in self.named_parameters()]
            return list_param
        elif stage=="maskim":
            list_param = [p for name, p in self.named_parameters() \
                    if (name.startswith("conv"))]
            return list_param
        else:
            warnings.warn("no mask param update")
            #raise NotImplementedError
            return []

    def forward(self, x, guide):
        xin = x
        bsize, ch, height, width = x.shape
        x = torch.cat([x, guide], 1)
        x_concat = x

        # two stage network
        ## stage1
        x = self.conv1(x_concat)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x_bneck = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample_conv(x)
        x = self.conv14(x)
        x = self.conv15_upsample_conv(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x_stage1 = torch.tanh(x)
        ## stage mask
        x = self.conv_mask_11(x_bneck)
        x = self.conv_mask_12(x)
        x = self.conv_mask_13_upsample_conv(x)
        x = self.conv_mask_14(x)
        x = self.conv_mask_15_upsample_conv(x)
        x = self.conv_mask_16(x)
        x = self.conv_mask_17(x)
        mask1 = torch.sigmoid(x)
        return mask1, x_stage1

if __name__ == "__main__":
    pass
