import pdb
import math
import torch
import models.networks as networks
import util.util as util
import random
import numpy as np
from models.create_mask import MaskCreator
from models.comod_model import CoModModel


class CoModIJ2Model(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        CoModModel.modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument('--load_pretrained_mask', type=str, required=False, help='load pt g')
            parser.add_argument('--load_pretrained_g_ema', type=str, required=False, help='load pt g')
            parser.add_argument('--no_mask_update', action='store_true')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        #assert opt.netG.lower() in ['comodgan', 'condmodgan', 'condmodganne']
        if 'condmod' in opt.netG.lower():
            assert opt.no_g_reg
        self.truncation_mean = None

        self.device = torch.device("cuda") if self.use_gpu() \
                else torch.device("cpu")
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netG_ema, self.netD, self.netM = self.initialize_networks(opt)
        if opt.factor is not None:
            self.eigvec = torch.load(opt.factor)["eigvec"].to(self.device)
        # set loss functions
        if opt.isTrain:
            print(f"looad {opt.load_pretrained_mask}")
            if not opt.continue_train:
                self.netM = util.load_network_path(
                        self.netM, opt.load_pretrained_mask)
                if opt.load_pretrained_g is not None:
                    print(f"looad {opt.load_pretrained_g}")
                    self.netG = util.load_network_path(
                            self.netG, opt.load_pretrained_g)
                if opt.load_pretrained_g_ema is not None:
                    print(f"looad {opt.load_pretrained_g_ema}")
                    self.netG_ema = util.load_network_path(
                            self.netG_ema, opt.load_pretrained_g_ema)
                if opt.load_pretrained_d is not None:
                    print(f"looad {opt.load_pretrained_d}")
                    self.netD = util.load_network_path(
                            self.netD, opt.load_pretrained_d)
            self.mask_creator = MaskCreator(opt.path_objectshape_list, opt.path_objectshape_base)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
        if opt.truncation is not None:
            self.truncation_mean = self.mean_latent(4096)

    def accumulate(self, decay=0.999):
        par1 = dict(self.netG_ema.named_parameters())
        par2 = dict(self.netG.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

        # set loss functions

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, is_real_im=True, **kwargs):
        inputs, real_image, edge, mask, inputs_cc, mask_cc, mask_image, mask0, inputs0, mean_path_length = self.preprocess_input(data)
        visline = inputs*(1-edge)+torch.ones_like(inputs)*edge
        visline0 = data['image']*(1-edge)+torch.ones_like(inputs)*edge
        bsize = real_image.size(0)
        if mode == 'generator':
            g_loss, uc_image = self.compute_generator_loss(
                    inputs, real_image, edge, mask, inputs_cc,mask_cc,mask_image,mask0,inputs0,
                    is_real_im=is_real_im,is_mask_loss=data['mask_loss'])
            composed_image = uc_image*mask + inputs
            generated = {'fake':composed_image,
                    'mask':mask,
                    'maskim':mask_image,
                    'maskin':visline0,
                    'input_cc':inputs_cc,
                    'gt': real_image,
                    'input':visline}
            return g_loss, inputs, generated
        elif mode == 'dreal':
            d_loss = self.compute_discriminator_loss(
                real_image, fake_image=None, edge=edge,mask=mask, inputs_cc=inputs_cc,mask_cc=mask_cc)
            return d_loss
        elif mode == 'dfake':
            with torch.no_grad():
                _image, uc_image, _ = self.generate_fake(inputs,real_image,edge,mask, inputs_cc,mask_cc)
                composed_image = uc_image*mask + inputs
                composed_image = composed_image.detach()
                composed_image.requires_grad_()
            d_loss = self.compute_discriminator_loss(
                real_image=None, fake_image=composed_image, edge=edge,mask=mask,inputs_cc=inputs_cc,mask_cc=mask_cc)
            return d_loss
        elif mode == 'd_reg':
            d_regs = self.compute_discriminator_reg(real_image, edge,mask, inputs_cc, mask_cc)
            return d_regs
        elif mode == 'g_reg':
            g_regs, path_lengths, mean_path_length = self.compute_generator_reg(
                    inputs,
                    real_image,
                    edge,
                    mask,
                    mean_path_length)
            return g_regs, mean_path_length
        elif mode == 'inference':
            with torch.no_grad():
                if self.opt.factor is None:
                    _image, uc_image,  _ = self.generate_fake(inputs,real_image,edge,mask, inputs_cc,mask_cc, ema=True)
                    composed_image = uc_image*mask + inputs
                else:
                    raise NotImplementedError
                    #fake_image, _ = self.factorize_fake(real_image, mask)
            return composed_image,mask 
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        M_params = list(self.netM.parameters())
        #if not opt.no_mask_update:
        #    G_params += M_params
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        g_reg_ratio = self.opt.g_reg_every / (self.opt.g_reg_every + 1)
        d_reg_ratio = self.opt.d_reg_every / (self.opt.d_reg_every + 1)

        g_optim = torch.optim.Adam(
                [
                    {"params":G_params},
                    {"params":M_params, "lr":2e-4, "betas":(self.opt.beta1,self.opt.beta2)}
                    ],
            lr=self.opt.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = torch.optim.Adam(
            D_params,
            lr=self.opt.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        return g_optim, d_optim

    def save(self, epoch):
        util.save_network(self.netM, 'M', epoch, self.opt)
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netG_ema, 'G_ema', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        _netg = opt.netG
        opt.netG="MD"
        netM = networks.define_G(opt)
        opt.netG = _netg
        netG_ema = networks.define_G(opt)
        if opt.isTrain:
            netG = networks.define_G(opt)
            netD = networks.define_D(opt)
        else:
            netD=None
            netG=None

        if not opt.isTrain or opt.continue_train:
            netG_ema = util.load_network(netG_ema, 'G_ema', opt.which_epoch, opt)
            netM = util.load_network(netM, 'M', opt.which_epoch, opt)
            if opt.isTrain:
                netG = util.load_network(netG, 'G', opt.which_epoch, opt)
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        return netG, netG_ema, netD, netM

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def mean_latent(self, n_latent):
        self.netG_ema.eval()
        latent_in = torch.randn(n_latent, self.opt.z_dim, device=self.device)
        dlatent = self.netG_ema(latents_in=[latent_in], get_latent=True)[0]
        latent_mean = dlatent.mean(0, keepdim=True)
        self.truncation_mean = latent_mean
        return self.truncation_mean

    def make_noise(self, batch, n_noise):
        if n_noise == 1:
            return torch.randn(batch, self.opt.z_dim, device=self.device)

        noises = torch.randn(n_noise, batch, self.opt.z_dim, 
                device=self.device).unbind(0)

        return noises

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

    def mixing_noise(self, batch):
        if self.opt.mixing > 0 and random.random() < self.opt.mixing:
            noise =  self.make_noise(batch, 2)
            return noise
        else:
            return [self.make_noise(batch, 1)]

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

        if 'mean_path_length' in data:
            mean_path_length = data['mean_path_length'].detach().cuda()
        else:
            mean_path_length = 0
        if not self.training or data['type'] == 'warp':
            edge = data['mask']
            # use estimated mask
            mask0, mask_image = self.netM(data['image'], edge)
            #flag = random.randint(0,2) if self.training else 2
            flag = 2
            if not self.training or flag==1 or flag==2:
                mask = mask0
                if not self.training or self.opt.no_mask_update:
                    mask = mask.detach()
                if flag==1:
                    mask = (mask>0.5).detach().float()
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
            #inputs = data['image']*(1-mask)
            mask0 = mask
            mask_image = data['image']
        else:
            raise NotImplementedError
        inputs0 = data['image']
        if not self.training:
            mask_cc = mask
        else:
            mask_cc = self.make_mask(data, size1=min(h,w)//8, size2=min(h,w)//4)
            mask_cc = (1-mask_cc)*mask
        inputs_cc = data['image']*mask_cc+torch.flip(data['image'],(0,))*mask*(1-mask_cc)
        mask_cc = mask
        inputs_cc = inputs_cc.detach()
        mask_cc = mask_cc.detach()
        if self.training:
            k = random.randint(-3,3)
            inputs_cc = torch.rot90(inputs_cc, k, (2,3))
            mask_cc = torch.rot90(mask_cc, k, (2,3))
            k = random.randint(0,1)
            if k > 0:
                inputs_cc = torch.flip(inputs_cc, (2,))
                mask_cc = torch.flip(mask_cc, (2,))
            k = random.randint(0,1)
            if k > 0:
                inputs_cc = torch.flip(inputs_cc, (3,))
                mask_cc = torch.flip(mask_cc, (3,))
        # inputs: G(inputs,mask,inputs_cc,mask_cc)
        # inputs0,mask0: for loss
        return inputs, data['gt'], data['edge'], data['mask'], inputs_cc, mask_cc, mask_image, mask0, inputs0, mean_path_length

    def g_path_regularize(self, fake_image, latents, mean_path_length, decay=0.01):
        noise = torch.randn_like(fake_image) / math.sqrt(
            fake_image.shape[2] * fake_image.shape[3]
        )
        grad, = torch.autograd.grad(
            outputs=(fake_image * noise).sum(), inputs=latents, create_graph=True
        )
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_mean.detach(), path_lengths

    def compute_generator_reg(self, inputs,real_image,edge,mask, mean_path_length):
        G_regs = {}
        bsize = real_image.size(0)
        path_batch_size = max(1, bsize // self.opt.path_batch_shrink)
        fake_image, _, latents = self.generate_fake(inputs,real_image,edge,mask, True)
        path_loss, mean_path_length, path_lengths = self.g_path_regularize(
            fake_image, latents, mean_path_length
        )
        weighted_path_loss = self.opt.path_regularize * self.opt.g_reg_every * path_loss

        if self.opt.path_batch_shrink:
            weighted_path_loss += 0 * fake_image[0, 0, 0, 0]
        G_regs['path'] = weighted_path_loss
        return G_regs, path_lengths, mean_path_length

    def g_image_loss(self,
            inputs,
            mask, mask_image,
            real_image, line):
        G_losses = {}
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

    def compute_generator_loss(self, inputs, real_image, edge, mask, inputs_cc,mask_cc,mask_image,mask0,inputs0,
            is_real_im=True, is_mask_loss=True):
        if is_mask_loss:
            G_mask_losses = self.g_image_loss(
                    inputs0,
                    mask0, mask_image,
                    real_image, edge)
            G_losses = {**G_mask_losses}
        else:
            G_losses = {}
        _image, uc_image, _ = self.generate_fake(
                inputs, real_image, edge, mask, inputs_cc,mask_cc)
        composed_image = uc_image*mask + inputs
        if is_real_im:
            pred_fake = self.netD(composed_image, edge, mask.detach(), inputs_cc.detach(), mask_cc.detach())

            G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)
            if not self.opt.no_vgg_loss:
                G_losses['VGG'] = self.criterionVGG(uc_image, real_image) \
                    * self.opt.lambda_vgg
            if not self.opt.no_l1_loss:
                G_losses['L1'] = torch.nn.functional.l1_loss(uc_image, real_image)  * self.opt.lambda_l1
        return G_losses, composed_image

    def compute_discriminator_reg(self, real_image, edge,mask,inputs_cc, mask_cc):
        real_image.requires_grad = True
        real_pred = self.netD(real_image, edge, mask.detach(), inputs_cc, mask_cc.detach())
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_image, create_graph=True
        )
        r1_loss = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        r1_loss = self.opt.r1 / 2 * r1_loss * self.opt.d_reg_every + 0 * real_pred[0]
        D_regs = {'r1': r1_loss}
        #for k,v in D_regs.items():
        #    D_regs[k] = v*0

        return D_regs

    def compute_discriminator_loss(self, real_image, fake_image=None, edge=None,mask=None, inputs_cc=None, mask_cc=None):
        D_losses = {}
        assert mask is not None
        assert fake_image is not None or real_image is not None
        assert fake_image is None or real_image is None
        if fake_image is not None:
            fake_image = fake_image.detach()
            pred_fake = self.netD(fake_image, edge,mask.detach(), inputs_cc, mask_cc.detach())
            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
        elif real_image is not None:
            pred_real = self.netD(real_image, edge, mask.detach(), inputs_cc, mask_cc.detach())
            D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                   for_discriminator=True)

        #for k,v in D_losses.items():
        #    D_losses[k] = v*0
        return D_losses

    def generate_fake(self,
            inputs,real_image,edge,mask, inputs_cc,mask_cc,
            return_latents=False, ema=False):
        bsize = real_image.size(0)
        noise = self.mixing_noise(bsize)
        if ema:
            self.netG_ema.eval()
            fake_image, uc_image, latent = self.netG_ema(
                    inputs.detach(),
                    edge,
                    mask.detach(),
                    inputs_cc,
                    mask_cc,
                    noise,
                    return_latents=return_latents,
                    truncation=self.opt.truncation,
                    truncation_latent=self.truncation_mean
                    )
        else:
            fake_image, uc_image, latent = self.netG(
                    inputs.detach(),
                    edge,
                    mask.detach(),
                    inputs_cc,
                    mask_cc,
                    noise, return_latents=return_latents)
        return fake_image, uc_image, latent

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

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
