import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize


class gen_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU()):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        super(gen_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        self.activation = activation

    def forward(self, x):
        x = super(gen_conv, self).forward(x)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

class gen_deconv(gen_conv):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv, self).forward(x)
        return x

class dis_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize=5, stride=2):
        """Define conv for discriminator.
        Activation is set to leaky_relu.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
        """
        p = int((ksize-1)/2)
        super(dis_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = super(dis_conv, self).forward(x)
        x = F.leaky_relu(x)
        return x

def batch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Define batch convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, out_channel, in_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)

    return out


def batch_transposeconv2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1):
    """Define batch transposed convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, in_channel, out_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, in_channels, out_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(in_channels*b_i, out_channels, kernel_height_size, kernel_width_size)

    out = F.conv_transpose2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding, output_padding=output_padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)
    return out



def hardmax(similar):
    val_max, id_max = torch.max(similar, 1)
    num = similar.size(1)
    sb = torch.Tensor(range(num)).long().to(similar.device)
    id_max = id_max[:, None, :, :]
    sb = sb[None, ..., None, None]
    similar = (sb==id_max).float().detach()
    return similar

class CP1(nn.Module):
    def __init__(self, bkg_patch_size=4, stride=1, ufstride=1, softmax_scale=10., nn_hard=False, pd=1,
                 fuse_k=3, is_fuse=False):
        super(CP1, self).__init__()
        self.bkg_patch_size = bkg_patch_size
        self.nn_hard = nn_hard
        self.stride = stride
        self.ufstride = ufstride
        self.softmax_scale = softmax_scale
        self.forward = self.forward_batch
        self.pd = pd
        self.fuse_k = fuse_k
        self.is_fuse = is_fuse

    def get_conv_kernel(self, x, mask=None):
        batch, c, h_small, w_small = x.shape
        x = x / torch.sqrt((x**2).sum(3, keepdim=True).sum(2, keepdim=True) + 1e-8)
        _x = F.pad(x, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        kernel = F.unfold(input=_x, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        kernel = kernel.transpose(1, 2) \
            .view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        # b*hw*c*k*c
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        m = m.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        m = m.squeeze(2)
        mm = (m.mean(3, keepdim=True).mean(2, keepdim=True)).float()
        #mm = (m.mean(3, keepdim=True).mean(2, keepdim=True)==1).float()
        return kernel, mm

    def forward_batch(self, f, b, mask=None):
        batch, c, h, w = b.shape
        batch, c, h_small, w_small = f.shape
        if mask is None:
            mask = torch.ones(batch, 1, h_small, w_small).to(f.device)
        else:
            mask = 1-mask
        # mask valid region
        softmax_scale = self.softmax_scale
        kernel, mmk = self.get_conv_kernel(b, mask)
        # mmk: valid ratio of each bkg patch
        _f = F.pad(f, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        cos_similar = batch_conv2d(_f, weight=kernel, stride=self.stride)
        _, cs, hs, ws = cos_similar.shape
        hb, wb = h//2, w//2

        if self.is_fuse:
            fuse_weight = torch.eye(self.fuse_k).to(f.device)
            fuse_weight = fuse_weight[None, None, ...]
            cos_similar = cos_similar.view(-1, cs, hs*ws)[:, None, ...]
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.reshape(batch, 1, cs, hs*ws)
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.squeeze(1)
            cos_similar = cos_similar.reshape(batch, cs, hs, ws)

        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), \
                     stride=self.stride)
        m = m.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        m = m.squeeze(2)
        mmp = (m.mean(3).mean(2)).float()
        mmp = mmp.view(batch, 1, hs, ws) # mmp: valid ratio of fg patch
        mm = (mmk>mmp).float()  # replace with more valid
        ppp = (mmp>0.5).float() # ppp: mask of partial valid
        mm = mm*ppp # partial valid being replaced with more valid
        mm = mm + (mmk==1).float().expand_as(mm)  # and full valid
        mm = (mm>0).float()
        cos_similar = cos_similar * mm
        cos_similar = F.softmax(cos_similar*softmax_scale, dim=1)
        if self.nn_hard:
            cos_similar = hardmax(cos_similar)
        return cos_similar

class CP2(nn.Module):
    def __init__(self, bkg_patch_size=16, stride=8, ufstride=8, pd=4):
        super(CP2, self).__init__()
        self.stride = stride
        self.bkg_patch_size = bkg_patch_size
        self.forward = self.forward_batch
        self.ufstride = ufstride
        self.pd = pd
        #self.forward = self.forward_test


    def get_deconv_kernel(self, b, mask):
        batch, c, h, w = b.shape
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        msk_kernel = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        msk_kernel = msk_kernel.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        _b = F.pad(b, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        bkg_kernel = F.unfold(input=_b, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        bkg_kernel = bkg_kernel.transpose(1, 2).view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        bkg_kernel = bkg_kernel*(1-msk_kernel)

        return bkg_kernel, msk_kernel

    def forward_batch(self, cos_similar, b, mask):
        # use original background for reconstruction
        _, _, hs, ws = cos_similar.shape
        bkg_kernel, msk_kernel = self.get_deconv_kernel(b, mask)
        #hard_similar = hardmax(cos_similar.detach())
        output = batch_transposeconv2d(cos_similar,
                                       weight=bkg_kernel,stride=self.stride)

        norm_kernel = torch.ones(1, 1, self.bkg_patch_size, self.bkg_patch_size).to(mask.device)
        weight_map = torch.ones(1, 1, hs, ws).to(mask.device)
        weight_map = F.conv_transpose2d(weight_map, norm_kernel, stride=self.stride)
        mask_recon = batch_transposeconv2d(cos_similar,
                                           weight=msk_kernel,stride=self.stride)
        mask_recon = mask_recon / weight_map
        output = output[:,:,self.pd:-self.pd,self.pd:-self.pd]
        #mask_recon = mask_recon[:,:,self.pd:-self.pd,self.pd:-self.pd]
        return output



if __name__ == "__main__":
    pass
