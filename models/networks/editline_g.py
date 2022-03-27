import torch
import random
import warnings
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize
from models.networks.base_network import BaseNetwork
from models.networks.utils import gen_conv, gen_deconv, dis_conv
from models.networks.splitcam import ReduceContextAttentionP1, ReduceContextAttentionP2


class DeepFillC2Generator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--use_cam', action='store_true',
                            help="use context attention module")
        parser.add_argument('--pool_type', default='max',
                            help="use context attention module")
        parser.add_argument('--no_mask_cc', action='store_true',
                            help="use context attention module")
        parser.add_argument('--no_mask_coarse', action='store_true',
                            help="use context attention module")

    def __init__(self, opt):
        super(DeepFillC2Generator, self).__init__()
        self.opt = opt
        self.pool_type = opt.pool_type
        self.use_cam = opt.use_cam
        cnum = 48
        self.cnum = cnum
        rate = 1
        self.cam_1 = ReduceContextAttentionP1(nn_hard=False, 
                ufstride=2*rate, 
                stride=2*rate, 
                bkg_patch_size=4*rate, pd=0, is_th=True, th=0.1, norm_type=1)
        self.cam_2 = ReduceContextAttentionP2(
                ufstride=2*rate, 
                bkg_patch_size=4*rate, 
                stride=2*rate, pd=0,mk=False)
        # stage1
        self.conv1 = gen_conv(5, cnum, 5, 1)
        self.conv2_downsample = gen_conv(int(cnum/2), 2*cnum, 3, 2)
        self.conv3 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.conv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.conv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.conv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.conv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16) #8
        self.conv11 = gen_conv(4*cnum, 4*cnum, 3, 1)
        self.conv12 = gen_conv(2*cnum, 4*cnum, 3, 1) #4
        self.conv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv14 = gen_conv(cnum, 2*cnum, 3, 1) #2
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum//2, cnum//2, 3, 1) #1
        self.conv17 = gen_conv(cnum//4, 3, 3, 1, activation=None)
        # stage1 warpin
        self.wconv1 = gen_conv(5, cnum, 5, 1)
        self.wconv2_downsample = gen_conv(int(cnum/2), 2*cnum, 3, 2)
        self.wconv3 = gen_conv(cnum, 2*cnum, 3, 1)
        self.wconv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.wconv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.wconv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.wconv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.wconv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.wconv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.wconv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16) #8

        # stage2
        self.xconv1 = gen_conv(3, cnum, 5, 1)
        self.xconv2_downsample = gen_conv(cnum//2, cnum, 3, 2)
        self.xconv3 = gen_conv(cnum//2, 2*cnum, 3, 1)
        self.xconv4_downsample = gen_conv(cnum, 2*cnum, 3, 2)
        self.xconv5 = gen_conv(cnum, 4*cnum, 3, 1)
        self.xconv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.xconv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.xconv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.xconv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.xconv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16)
        self.pmconv1 = gen_conv(3, cnum, 5, 1)
        self.pmconv2_downsample = gen_conv(cnum//2, cnum, 3, 2)
        self.pmconv3 = gen_conv(cnum//2, 2*cnum, 3, 1)
        self.pmconv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.pmconv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.pmconv6 = gen_conv(2*cnum, 4*cnum, 3, 1, 
                            activation=nn.ReLU())
        self.pmconv9 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.pmconv10 = gen_conv(2*cnum, 4*cnum, 3, 1)

        self.allconv11 = gen_conv(4*cnum, 4*cnum, 3, 1)
        self.allconv12 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.allconv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.allconv14 = gen_conv(cnum, 2*cnum, 3, 1)
        self.allconv15_upsample_conv = gen_deconv(cnum, cnum)
        self.allconv16 = gen_conv(cnum//2, cnum//2, 3, 1)
        self.allconv17 = gen_conv(cnum//4, 3, 3, 1, activation=None)

    def get_param_list(self, stage="all"):
        if stage=="all" or stage=="image":
            list_param = [p for name, p in self.named_parameters()]
            return list_param
        elif stage=="coarse":
            list_param = [p for name, p in self.named_parameters() \
                    if (name.startswith("conv"))]
            return list_param
        elif stage=="fine":
            list_param = [p for name, p in self.named_parameters() \
                    if not (name.startswith("conv"))]
            return list_param
        else:
            warnings.warn("no generator param update")
            return []


    def forward(self, x, x2, mask, mask2, guide=None, guide2=None):
        x2 = x2*mask2
        x = x*(1-mask)
        xin = x
        bsize, ch, height, width = x.shape
        if guide is None:
            ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        else:
            ones_x = guide
        ones_x2 = torch.ones(bsize, 1, height, width).to(x.device)
        #if guide2 is None:
        #    ones_x2 = torch.ones(bsize, 1, height, width).to(x.device)
        #else:
        #    ones_x2 = guide2
        x = torch.cat([x, ones_x, mask], 1)
        x2 = torch.cat([x2, ones_x2, mask2], 1)
        # two stage network
        ## stage1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)

        x2 = self.wconv1(x2)
        x2 = self.wconv2_downsample(x2)
        x2 = self.wconv3(x2)
        x2 = self.wconv4_downsample(x2)
        x2 = self.wconv5(x2)
        x2 = self.wconv6(x2)
        x2 = self.wconv7_atrous(x2)
        x2 = self.wconv8_atrous(x2)
        x2 = self.wconv9_atrous(x2)
        x2 = self.wconv10_atrous(x2)
        _,_,hs,ws = x2.shape
        if self.pool_type == 'avg':
            x2 = x2.mean(3).mean(2)[...,None,None]
        elif self.pool_type == 'max':
            x2 = F.max_pool2d(x2, kernel_size=(hs, ws))
        else:
            raise NotImplementedError
        x2 = F.interpolate(x2, (hs,ws), mode='nearest')
        x = torch.cat((x,x2),1)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample_conv(x)
        x = self.conv14(x)
        x = self.conv15_upsample_conv(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = torch.tanh(x)
        x_stage1 = x

        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)
        xnow = x

        ###
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        ###
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        if self.use_cam:
            mask_s = F.avg_pool2d(mask, kernel_size=4, stride=4)
            mask_s = mask_s.detach()
            similar = self.cam_1(x, x, mask_s)
            x, recon_aux = self.cam_2(similar, x, mask_s, {})
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], 1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample_conv(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample_conv(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.tanh(x)
        return x_stage1, x_stage2


if __name__ == "__main__":
    pass
