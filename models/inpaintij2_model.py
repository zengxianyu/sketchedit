"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import pdb
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import models.networks as networks
import util.util as util
from models.create_mask import MaskCreator
import random
import numpy as np

# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=kernel_size//2, padding_mode='replicate')

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter



class InpaintIJ2Model(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        parser.add_argument('--no_mask_update', action='store_true')
        parser.add_argument('--no_train0', action='store_true')
        parser.add_argument('--th_mask', action='store_true')
        parser.add_argument('--filt_im', action='store_true')
        parser.add_argument('--masked_in', action='store_true')
        parser.add_argument('--detach_in', action='store_true')
        parser.add_argument('--load_pretrained_mask', type=str, required=False, help='load pt g')
        parser.add_argument('--load_pretrained_g', type=str, required=False, help='load pt g')
        parser.add_argument('--load_pretrained_d', type=str, required=False, help='load pt d')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netM = self.initialize_networks(opt)
        self.filter = get_gaussian_kernel(kernel_size=3, sigma=2, channels=3)
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.filter.cuda()
        if opt.isTrain:
            if not opt.continue_train:
                print(f"looad {opt.load_pretrained_mask}")
                self.netM = util.load_network_path(
                        self.netM, opt.load_pretrained_mask)
                if opt.load_pretrained_g is not None:
                    print(f"looad {opt.load_pretrained_g}")
                    self.netG = util.load_network_path(
                            self.netG, opt.load_pretrained_g)
                if opt.load_pretrained_d is not None:
                    print(f"looad {opt.load_pretrained_d}")
                    self.netD = util.load_network_path(
                            self.netD, opt.load_pretrained_d)

        # set loss functions
        if opt.isTrain:
            self.mask_creator = MaskCreator(opt.path_objectshape_list, opt.path_objectshape_base)
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, is_real_im=True, **kwargs):
        inputs, real_image, edge, mask, inputs_cc, mask_cc, edge_cc, mask_image, mask0, inputs0 \
                = self.preprocess_input(data)
        visline = inputs*(1-edge)+torch.ones_like(inputs)*edge
        visline0 = data['image']*(1-edge)+torch.ones_like(inputs)*edge

        if mode == 'generator':
            g_loss, coarse_image, composed_image = \
                    self.compute_generator_loss(
                    inputs, real_image, edge, mask,
                    inputs_cc,mask_cc,edge_cc,mask_image,mask0,inputs0,
                    is_real_im=is_real_im,mask_loss=data['mask_loss'])
            generated = {'fake':composed_image,
                    'mask':mask,
                    'maskim':mask_image,
                    'maskin':visline0,
                    'input_cc':inputs_cc,
                    'gt': real_image,
                    'input':visline}
            return g_loss, inputs, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                inputs, real_image, edge, mask, inputs_cc, mask_cc,edge_cc)
            return d_loss, data['inputs']
        elif mode == 'inference':
            with torch.no_grad():
                coarse_image, fake_image = self.generate_fake(inputs, real_image, edge, mask, inputs_cc, mask_cc, edge_cc)
                composed_image = fake_image*mask + inputs
            return composed_image,mask
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = self.netG.get_param_list()
        M_params = list(self.netM.parameters())
        if not opt.no_mask_update:
            G_params += M_params
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netM, 'M', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        _netg = opt.netG
        opt.netG="MD"
        netM = networks.define_G(opt)
        opt.netG = _netg
        netG = networks.define_G(opt)
        if opt.isTrain:
            netD = networks.define_D(opt)
        else:
            netD=None

        if not opt.isTrain or opt.continue_train:
            netM = util.load_network(netM, 'M', opt.which_epoch, opt)
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        return netG, netD, netM

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def make_mask(self, data, size1=None, size2=None):
        b,c,h,w = data['image'].shape
        if size1 is None:
            size1 = min(h,w)//4
        if size2 is None:
            size2 = min(h,w)//2
        if self.opt.isTrain:
            # generate random stroke mask
            mask1 = self.mask_creator.stroke_mask(h, w, max_length=size2)
            # generate object/square mask
            ri = random.randint(0,3)
            if not self.opt.not_om and (ri  == 1 or ri == 0):
                mask2 = self.mask_creator.object_mask(h, w)
            else:
                mask2 = self.mask_creator.rectangle_mask(h, w, 
                        size1, size2)
            # use the mix of two masks
            mask = (mask1+mask2>0)
            mask = mask.astype(np.float)
            mask = self.FloatTensor(mask)[None, None,...].expand(b,-1,-1,-1)
        else:
            mask = mask.cuda()
        return mask

    def preprocess_input(self, data):
        b,c,h,w = data['image'].shape
        if self.use_gpu():
            data['image'] = data['image'].cuda()
            if 'mask' in data:
                data['mask'] = data['mask'].cuda()
            if 'gt' in data:
                data['gt'] = data['gt'].cuda()
            else:
                data['gt'] = data['image'].cuda()
            if 'edgegt' in data:
                data['edgegt'] = data['edgegt'].cuda()

        if not self.training or data['type'] == 'warp':
            edge = data['mask']
            # use estimated mask
            mask0, mask_image = self.netM(data['image'], edge)
            #flag = random.randint(0,2) if self.training else 2
            if self.opt.th_mask:
                flag = 1
            else:
                flag = 2
            if not self.training or flag==1 or flag==2:
                mask = mask0
                if self.opt.no_mask_update:
                    mask = mask.detach()
                if flag==1:
                    mask = (mask>0.95).detach().float()
                data['mask'] = mask
                data['edge'] = edge
                inputs = data['image']*(1-mask)
        elif data['type'] == 'inpaint':
            data['mask_loss'] = False
            edgegt = data['edgegt']
            mask = self.make_mask(data)
            data['mask'] = mask
            data['edge'] =  edgegt*mask
            inputs = data['gt']*(1-mask)
            mask0 = mask
            mask_image = data['image']
        else:
            raise NotImplementedError
        inputs0 = data['image']
        if not self.training:
            mask_cc = mask
        else:
            mask_cc = self.make_mask(data)
            mask_cc = (1-mask_cc)*mask
        if self.opt.masked_in:
            inputs_cc = data['image']*mask_cc
        else:
            inputs_cc = data['image']*mask_cc+torch.flip(data['image'],(0,))*mask*(1-mask_cc)
            mask_cc = mask
        edge_cc = data['edge']
        #if self.training:
        #    k = random.randint(-3,3)
        #    inputs_cc = torch.rot90(inputs_cc, k, (2,3))
        #    mask_cc = torch.rot90(mask_cc, k, (2,3))
        #    edge_cc = torch.rot90(data['edge'], k, (2,3))
        #    k = random.randint(0,1)
        #    if k > 0:
        #        inputs_cc = torch.flip(inputs_cc, (2,))
        #        mask_cc = torch.flip(mask_cc, (2,))
        #        edge_cc = torch.flip(data['edge'], (2,))
        #    k = random.randint(0,1)
        #    if k > 0:
        #        inputs_cc = torch.flip(inputs_cc, (3,))
        #        mask_cc = torch.flip(mask_cc, (3,))
        #        edge_cc = torch.flip(data['edge'], (3,))
        #else:
        #    edge_cc = data['edge']
        # inputs: G(inputs,mask,inputs_cc,mask_cc)
        # inputs0,mask0: for loss
        data['inputs'] = inputs
        return inputs, data['gt'], data['edge'], data['mask'], inputs_cc, mask_cc, edge_cc, mask_image, mask0, inputs0

    def g_image_loss(self,
            inputs,
            mask, mask_image,
            real_image, line):
        G_losses = {}
        if self.opt.filt_im:
            real_image = self.filter(real_image)
            inputs = self.filter(inputs)
        in_ims = {'mask':inputs}
        out_ims = {'mask':mask_image}
        com_masks = {'mask':mask}
        com_ims = {}
        for k,v in out_ims.items():
            com_ims[k] = out_ims[k]*com_masks[k]+in_ims[k]*(1-com_masks[k])
        G_losses['L1c'] = 0
        G_losses['L1c'] += (torch.nn.functional.l1_loss(out_ims['mask'], real_image)\
                * self.opt.lambda_l1_mask)
        G_losses['L1c'] += (torch.nn.functional.l1_loss(com_ims['mask'], real_image)\
                * self.opt.lambda_l1_mask)
        return G_losses


    def compute_generator_loss(self, inputs, real_image, edge, mask,
            inputs_cc,mask_cc,edge_cc,mask_image,mask0,inputs0,is_real_im=True,mask_loss=True):
        if mask_loss:
            G_mask_losses = self.g_image_loss(
                    inputs0,
                    mask0, mask_image,
                    real_image, edge)
            G_losses = {**G_mask_losses}
        else:
            G_losses = {}

        coarse_image, fake_image = self.generate_fake(
            inputs, real_image, edge, mask, inputs_cc, mask_cc, edge_cc)

        composed_image = fake_image*mask + inputs
        if is_real_im:
            if not self.opt.no_gan_loss:
                pred_fake, pred_real = self.discriminate(
                    inputs, composed_image, real_image, edge, mask)

                G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                    for_discriminator=False)
            if not self.opt.no_vgg_loss:
                G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                    * self.opt.lambda_vgg
            if not self.opt.no_l1_loss:
                #_G_losses = self.g_image_loss(inputs, mask, fake_image, real_image, edge)
                #for k,v in _G_losses.items(): G_losses[k] += v
                #_G_losses = self.g_image_loss(inputs, mask, coarse_image, real_image, edge)
                #for k,v in _G_losses.items(): G_losses[k] += v
                G_losses['L1'] = torch.nn.functional.l1_loss(fake_image, real_image)  * self.opt.lambda_l1
                G_losses['L1'] += torch.nn.functional.l1_loss(coarse_image, real_image)  * self.opt.lambda_l1
                if mask_loss:
                    if not self.opt.no_vgg_loss:
                        G_losses['VGG'] += self.criterionVGG(composed_image, real_image) \
                            * self.opt.lambda_vgg
                    G_losses['L1'] += torch.nn.functional.l1_loss(composed_image, real_image)  * self.opt.lambda_l1
        return G_losses, coarse_image, composed_image

    def compute_discriminator_loss(self, inputs, real_image, edge, mask, inputs_cc, mask_cc, edge_cc):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                coarse_image, fake_image = self.generate_fake(inputs, real_image, edge, mask,
                        inputs_cc, mask_cc, edge_cc)
                fake_image = fake_image.detach()
                fake_image.requires_grad_()
                composed_image = fake_image*mask + inputs

            pred_fake, pred_real = self.discriminate(
                inputs, composed_image, real_image, edge, mask)

            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                   for_discriminator=True)

        return D_losses

    def generate_fake(self, inputs, real_image, edge, mask, inputs_cc, mask_cc, edge_cc):
        if self.opt.detach_in:
            coarse_image, fake_image = self.netG(inputs.detach(), inputs_cc.detach(), mask.detach(),
                    mask_cc.detach(), edge.detach(), edge_cc.detach())
        else:
            coarse_image, fake_image = self.netG(inputs, inputs_cc, mask,
                    mask_cc, edge, edge_cc)

        return coarse_image, fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, inputs, fake_image, real_image, edge, mask):
        #if self.opt.detach_in:
        #    fake_image = fake_image*mask.detach()+inputs.detach()
        #else:
        #    fake_image = fake_image*mask.detach()+real_image*(1-mask.detach())
        fake_image = fake_image*mask.detach()+real_image*(1-mask.detach())
        fake_concat = fake_image
        real_concat = real_image

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        edge_merge = torch.cat((edge, edge),0)
        discriminator_out = self.netD(fake_and_real, edge_merge.detach())

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
