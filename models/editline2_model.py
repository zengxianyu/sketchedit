import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import models.networks as networks
from models.create_mask import MaskCreator
import util.util as util
import random
import numpy as np
import pdb

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


class EditLine2Model(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        #parser.add_argument('--mim_in', action='store_true', help='')
        if is_train:
            parser.add_argument('--update_part', type=str, default='all', help='update part')
            parser.add_argument('--load_pretrained_mask', type=str, required=False, help='load pt g')
            parser.add_argument('--load_pretrained_g', type=str, required=False, help='load pt g')
            parser.add_argument('--load_pretrained_d', type=str, required=False, help='load pt d')
            parser.add_argument('--filt_maskim', action='store_true', help='')
            #parser.add_argument('--gt_compose', action='store_true', help='')
            #parser.add_argument('--rbz_mask', action='store_true', help='')
            parser.add_argument('--no_detach', action='store_true', help='')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netM, self.netG, self.netD = self.initialize_networks(opt)
        self.filter = get_gaussian_kernel(kernel_size=3, sigma=2, channels=3)
        if self.use_gpu():
            self.filter = self.filter.cuda()
        if opt.isTrain:
            self.mask_creator = MaskCreator(opt.path_objectshape_list, opt.path_objectshape_base)
        if opt.isTrain and opt.load_pretrained_mask is not None:
            print(f"looad {opt.load_pretrained_mask}")
            self.netM = util.load_network_path(
                    self.netM, opt.load_pretrained_mask)
        if opt.isTrain and opt.load_pretrained_g is not None:
            print(f"looad {opt.load_pretrained_g}")
            self.netG = util.load_network_path(
                    self.netG, opt.load_pretrained_g)
        if opt.isTrain and opt.load_pretrained_d is not None:
            print(f"looad {opt.load_pretrained_d}")
            self.netD = util.load_network_path(
                    self.netD, opt.load_pretrained_d)

        # set loss functions
        if opt.isTrain:
            self.train_mask = False
            self.train_maskim = False
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, is_real_im=True):
        inputs, real_image, line, line_full, random_mask = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, coarse_image,fake_image, mask, mask_image, mask_inpaint, line_inpaint, input_inpaint = self.compute_generator_loss(
                inputs, real_image, line, line_full, random_mask, is_real_im=is_real_im)
            visline = input_inpaint*(1-line_inpaint)+torch.ones_like(input_inpaint)*line_inpaint
            composed_image = fake_image*mask_inpaint+input_inpaint*(1-mask_inpaint)
            generated = {
                    'mask': mask,
                    'maskim':mask_image,
                    'visline': visline,
                    'coarse':coarse_image,
                    'composed':composed_image,
                    'gt': real_image
                    }
            return g_loss, inputs, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                inputs, real_image, line, line_full, random_mask)
            return d_loss, data['image']
        elif mode == 'inference':
            with torch.no_grad():
                coarse_image, fake_image, mask, mask_image, mask_inpaint, line_inpaint, input_inpaint = self.generate_fake(
                        inputs, real_image, line, line_full, random_mask)
                composed_image = fake_image*mask + inputs*(1-mask)
            return composed_image, mask
        elif mode == 'visualize':
            with torch.no_grad():
                coarse_image, fake_image, mask, mask_image, mask_inpaint, line_inpaint, input_inpaint = self.generate_fake(
                        inputs, real_image, line, line_full, random_mask)
                composed_image = fake_image*mask + inputs*(1-mask)
                generated = {
                        'mask':mask_inpaint,
                        'maskim':mask_image,
                        'coarse':coarse_image,
                        'fine':fake_image,
                        'composed':composed_image}
            return generated
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params_mask = self.netM.get_param_list(opt.update_part)
        G_params_gen = self.netG.get_param_list(opt.update_part)
        G_params = G_params_mask + G_params_gen
        if len(G_params_gen)==0:
            self.train_mask = True
            print("====================train mask estimator=======================")
        if opt.update_part == 'maskim':
            self.train_maskim = True
            print("====================train mask estimator half=======================")
        #G_params = [p for name, p in self.netG.named_parameters() \
        #        if (not name.startswith("coarse"))]
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
        util.save_network(self.netM, 'M', epoch, self.opt)
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        _netg = opt.netG
        opt.netG="MD"
        netM = networks.define_G(opt)
        opt.netG = _netg
        if opt.isTrain:
            netD = networks.define_D(opt)
        else:
            netD=None

        if (not opt.isTrain and (not hasattr(opt,'isSkip'))) or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            netM = util.load_network(netM, 'M', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        return netM, netG, netD

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def get_external_mask(self, data):
        b,c,h,w = data.shape
        # generate random stroke mask
        mask1 = self.mask_creator.stroke_mask(h, w, max_length=min(h,w)/2)
        # generate object/square mask
        ri = random.randint(0,3)
        if not self.opt.not_om and (ri  == 1 or ri == 0):
            mask2 = self.mask_creator.object_mask(h, w)
        else:
            mask2 = self.mask_creator.rectangle_mask(h, w, 
                    min(h,w)//4, min(h,w)//2)
        # use the mix of two masks
        mask = (mask1+mask2>0)
        mask = mask.astype(np.float)
        mask = self.FloatTensor(mask)[None, None,...].expand(b,-1,-1,-1)
        return mask

    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            data['image'] = data['image'].cuda()
            if 'gt' in data:
                data['gt'] = data['gt'].cuda()
            else:
                data['gt'] = data['image'].cuda()
            data['mask'] = data['mask'].cuda()
            if 'edgegt' in data:
                data['edgegt'] = data['edgegt'].cuda()
            else:
                data['edgegt'] = data['mask'].cuda()
        if self.opt.isTrain:
            random_mask = self.get_external_mask(data['image'])
            if self.use_gpu():
                random_mask = random_mask.cuda()
        else:
            random_mask = None
        return data['image'], data['gt'], data['mask'], data['edgegt'], random_mask

    def g_image_loss(self,
            inputs, input_inpaint,
            coarse_image, fake_image,mask_inpaint,
            mask, mask_image,
            real_image, line, line_inpaint, is_real_im=True):
        if self.opt.filt_maskim:
            real_image_blur = self.filter(real_image)
            inputs_blur = self.filter(inputs)
            input_inpaint_blur = self.filter(input_inpaint)
        else:
            real_image_blur = real_image
            inputs_blur = inputs
            input_inpaint_blur = input_inpaint
        G_losses = {}
        blur_in_ims = {'coarse':input_inpaint_blur,'fake':input_inpaint_blur,'mask':inputs_blur}
        in_ims = {'coarse':input_inpaint,'fake':input_inpaint,'mask':inputs}
        out_ims = {'coarse':coarse_image,'fake':fake_image,'mask':mask_image}
        com_masks = {'coarse':mask_inpaint,'fake':mask_inpaint,'mask':mask}
        com_ims = {}
        blur_out_ims = {}
        blur_com_ims = {}
        for k,v in out_ims.items():
            if self.opt.filt_maskim:
                blur_out_ims[k] = self.filter(out_ims[k])
            else:
                blur_out_ims[k] = out_ims[k]
            blur_com_ims[k] = out_ims[k]*com_masks[k]+blur_in_ims[k]*(1-com_masks[k])
            com_ims[k] = out_ims[k]*com_masks[k]+in_ims[k]*(1-com_masks[k])
        if not self.train_mask and not self.opt.no_gan_loss and is_real_im:
            pred_fake, pred_real = self.discriminate(
                com_ims['fake'], real_image, line_inpaint, inputs, mask_inpaint)

            G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            raise NotImplementedError

        if not self.train_mask and not self.opt.no_vgg_loss:
            if is_real_im:
                G_losses['VGG'] = \
                        (self.criterionVGG(out_ims['fake'], real_image)*self.opt.lambda_vgg)
        G_losses['L1c'] = 0
        if not self.train_mask and is_real_im:
            G_losses['L1c'] = \
                    (torch.nn.functional.l1_loss(out_ims['coarse'],real_image)\
                    )* self.opt.lambda_l1
            if self.opt.update_part=="all" or self.opt.update_part=="fine":
                G_losses['L1f'] = \
                        (torch.nn.functional.l1_loss(out_ims['fake'], real_image)\
                        )* self.opt.lambda_l1
        G_losses['L1c'] += (torch.nn.functional.l1_loss(out_ims['mask'], real_image_blur)\
                * self.opt.lambda_l1_mask)
        if not self.train_maskim:
            G_losses['L1c'] += (torch.nn.functional.l1_loss(blur_com_ims['mask'], real_image_blur)\
                    * self.opt.lambda_l1_mask)
        #G_losses['mask'] = torch.nn.functional.l1_loss(mask, (mask>0.5).float())*0.001
        return G_losses


    def compute_generator_loss(self, inputs, real_image, line, line_full, random_mask, is_real_im=True):

        coarse_image, fake_image, mask, mask_image, mask_inpaint, line_inpaint, input_inpaint = self.generate_fake(
            inputs, real_image, line, line_full, random_mask)

        G_losses = self.g_image_loss(
                inputs, input_inpaint,
                coarse_image, fake_image,mask_inpaint,
                mask, mask_image,
                real_image, line, line_inpaint, is_real_im=is_real_im)


        return G_losses, coarse_image,fake_image, mask, mask_image, mask_inpaint, line_inpaint, input_inpaint

    def compute_discriminator_loss(self, inputs, real_image, line, line_full, random_mask):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                coarse_image, fake_image, mask, mask_image, mask_inpaint, line_inpaint, input_inpaint = self.generate_fake(
                    inputs, real_image, line, line_full, random_mask)
                composed_image = fake_image*mask_inpaint + input_inpaint*(1-mask_inpaint)
                composed_image = composed_image.detach()
                composed_image.requires_grad_()

            pred_fake, pred_real = self.discriminate(
                composed_image, real_image, line_inpaint, inputs, mask_inpaint)

            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                   for_discriminator=True)

        return D_losses

    def generate_fake(self, inputs, real_image, line, line_full, random_mask):
        mask, mask_image = self.netM(inputs, line)
        if self.opt.joint_train_inp:
            flag =  random.randint(0,2)
        else:
            flag =  random.randint(1,2)
        inputs0 = inputs
        if not self.training:
            mask_inpaint = mask.detach()
            mask_inpaint = (mask_inpaint>0.5).float().detach()
            line_inpaint = line
        elif flag == 0:
            mask_inpaint = random_mask
            line_inpaint = line_full*mask_inpaint
            inputs0 = real_image
        elif flag == 1:
            mask_inpaint = mask
            if not self.opt.no_detach:
                mask_inpaint = mask_inpaint.detach()
            line_inpaint = line
        elif flag == 2:
            mask_inpaint = mask
            mask_inpaint = (mask_inpaint>0.5).float().detach()
            line_inpaint = line
        if self.training:
            rm2 = self.get_external_mask(inputs)
            rm2 = (1-rm2.to(mask_inpaint.device))*mask_inpaint
        else:
            rm2 = mask_inpaint

        coarse_image, fake_image = self.netG(inputs0, inputs, mask_inpaint, rm2, line_inpaint)

        return coarse_image, fake_image, mask, mask_image, mask_inpaint, line_inpaint, inputs0

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, fake_image, real_image, line, inputs, mask):
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.

        fake_image = fake_image*mask.detach()+real_image*(1-mask.detach())
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        line_cat = torch.cat([line, line], dim=0)
        cc_cat = torch.cat([inputs, inputs], dim=0)

        discriminator_out = self.netD(fake_and_real, line_cat, cc=cc_cat)

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
