import torch
import os
import math
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# from torch.utils.serialization import load_lua
from lib.nn import SynchronizedBatchNorm2d as SynBN2d

###############################################################################
# Functions
###############################################################################

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16
# '''前面这部分是对称裁切'''
    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0
#  '''对称翻转填充'''
        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'synBN':
        norm_layer = functools.partial(SynBN2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False, opt=None):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

         #'''走的这个'''
    if which_model_netG == 'sid_unet_resize':
        netG = Unet_resize_conv(opt, skip)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) >= 0:
        netG.cuda(device=gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'no_norm':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'no_norm_4':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'no_patchgan':
        netD = FCDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, patch=patch)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes LOSS
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        # '''1为真，0为假'''
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
#         self.TV = L_TV ()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
#         self.loss = self.loss + self.TV   #自填
        return self.loss(input, target_tensor)



class DiscLossWGANGP():
    def __init__(self):
        self.LAMBDA = 10
        
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
       
        self.LAMBDA = 10
        
   
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        
        disc_interpolates = netD.forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

class SkipModule(nn.Module):
    def __init__(self, submodule, opt):
        super(SkipModule, self).__init__()
        self.submodule = submodule
        self.opt = opt

    def forward(self, x):
        latent = self.submodule(x)
        return self.opt.skip*x + latent, latent

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
       # 512-512
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        #512-1

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
   ## 模型容易，依次执行
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)

class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma /(embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma /(torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

# 颜色一致性损失
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        m = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return m
# TV-loss
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1.0):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
class Unet_resize_conv(nn.Module):
    def __init__(self, opt, skip):
        super(Unet_resize_conv, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
      
        if opt.self_attention:
            self.gate0 = GCT(4)
            self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
            self.shortcut1 = nn.Conv2d(4,32,3, padding=p)
            # self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
            #  """这里的几个池化层主要是对灰度图（自正则图）进行下采样"""
            self.downsample_1 = nn.MaxPool2d(2)
            self.downsample_2 = nn.MaxPool2d(2)
            self.downsample_3 = nn.MaxPool2d(2)
            self.downsample_4 = nn.MaxPool2d(2)
        else:
            
            self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
#        
        if self.opt.use_norm == 1:
            self.bn1_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.gate1 = GCT(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)   # 
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.max_pool1 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)
        self.gate2 = GCT(32)
        self.shortcut2 = nn.Conv2d(32,64,3, padding=p)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)    # 512*512*64
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.gate3 = GCT(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)    # 512*512*64
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.max_pool2 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)  #512*512*128
        self.gate4 = GCT(64)
        self.shortcut3 = nn.Conv2d(64,128,3, padding=p)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.gate5 = GCT(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)                                    #  128*128*128
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.max_pool3 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2) 
        self.gate6 = GCT(128)
        self.shortcut4 = nn.Conv2d(128,256,3, padding=p)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.gate7 = GCT(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.max_pool4 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)
        self.gate8 = GCT(256)
        self.shortcut5 = nn.Conv2d(256,512,3, padding=p)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
    
        if self.opt.use_norm == 1:
            self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
          
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
      
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        #self.gate6 = GCT(512)
        self.shortcut6 = nn.Conv2d(512,256,3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
    
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        #self.gate7 = GCT(256)
        self.shortcut7 = nn.Conv2d(256,128,3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
       # self.gate8 = GCT(128)
        self.shortcut8 = nn.Conv2d(128,64,3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
       # self.gate9 = GCT(64)
        self.shortcut9 = nn.Conv2d(64,32,3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
    
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        #self.gate10 = GCT(32)
        self.conv10 = nn.Conv2d(32, 3, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size*block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).resize(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input, gray):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
#         
        if self.opt.self_attention:
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
        if self.opt.use_norm == 1:
#        
            if self.opt.self_attention:
                x=torch.cat((input, gray), 1)
                x1 = x
                x = self.gate0(x)
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(x)))
           
                # x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            else:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            x = self.gate1(x)
            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x))) 
            conv1 = self.shortcut1(x1) + conv1
            x = self.max_pool1(conv1)
        
            x = self.gate2(x)
            x1 = x
            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            x = self.gate3(x)
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x))) 
            conv2 = self.shortcut2(x1) + conv2
            
            x = self.max_pool2(conv2) 
            x = self.gate4(x)
            x1 = x
            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            x = self.gate5(x)
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x))) 
            conv3 = self.shortcut3(x1) + conv3
            x = self.max_pool3(conv3)
            x = self.gate6(x)
            x1 = x
            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            x = self.gate7(x)
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x))) 
            conv4 = self.shortcut4(x1) + conv4
            x = self.max_pool4(conv4)
        
            x = self.gate8(x)
            x1 = x
            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            
            x = x*gray_5 if self.opt.self_attention else x
    
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
            conv5 = self.shortcut5(x1) + conv5
            
            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear') 
            conv4 = conv4*gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
           
            up = up6
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))
            conv6 = self.shortcut6(up) + conv6

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear') 
            conv3 = conv3*gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
          
            up = up7
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))
            conv7 = self.shortcut7(up) + conv7

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear') 
            conv2 = conv2*gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
           
            up = up8
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))
            conv8 = self.shortcut8(up) + conv8

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear') 
            conv1 = conv1*gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            #up9 = self.gate9(up9)
            up = up9
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
           # x = self.gate20(x)
            conv9 = self.LReLU9_2(self.conv9_2(x)) 
            conv9 = self.shortcut9(up) + conv9            #x = self.gate10(x)

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent*gray
  # 下述代码很多没有执行
            # output = self.depth_to_space(conv10, 2)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    input = (input - torch.min(input))/(torch.max(input) - torch.min(input))
                    output = latent + input*self.opt.skip
                    output = output*2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
#                         '''就执行了这一句'''
                    output = latent + input*self.opt.skip
            else:
                output = latent

                
        elif self.opt.use_norm == 0:
            if self.opt.self_attention:
                x = self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1)))
            else:
                x = self.LReLU1_1(self.conv1_1(input))
            conv1 = self.LReLU1_2(self.conv1_2(x))
            x = self.max_pool1(conv1)

            x = self.LReLU2_1(self.conv2_1(x))
            conv2 = self.LReLU2_2(self.conv2_2(x))
            x = self.max_pool2(conv2)

            x = self.LReLU3_1(self.conv3_1(x))
            conv3 = self.LReLU3_2(self.conv3_2(x))
            x = self.max_pool3(conv3)

            x = self.LReLU4_1(self.conv4_1(x))
            conv4 = self.LReLU4_2(self.conv4_2(x))
            x = self.max_pool4(conv4)

            x = self.LReLU5_1(self.conv5_1(x))
            x = x*gray_5 if self.opt.self_attention else x
            conv5 = self.LReLU5_2(self.conv5_2(x))

            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4*gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.LReLU6_1(self.conv6_1(up6))
            conv6 = self.LReLU6_2(self.conv6_2(x))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3*gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.LReLU7_1(self.conv7_1(up7))
            conv7 = self.LReLU7_2(self.conv7_2(x))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2*gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.LReLU8_1(self.conv8_1(up8))
            conv8 = self.LReLU8_2(self.conv8_2(x))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1*gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.LReLU9_1(self.conv9_1(up9))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)
            
            if self.opt.times_residual:
                latent = latent*gray

            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    input = (input - torch.min(input))/(torch.max(input) - torch.min(input))
                    output = latent + input*self.opt.skip
                    output = output*2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    output = latent + input*self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output/torch.max(torch.abs(output))
        
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        if self.skip:

            return output, latent
        else:
            return output


class DnCNN(nn.Module):
    def __init__(self, opt=None, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y+out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
# ''' VGG16网络'''
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2) 
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3

def vgg_preprocess(batch, opt):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean)) # subtract mean
    return batch

class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        if self.opt.no_vgg_instance:
            return torch.mean((img_fea - target_fea) ** 2)
        else:
            return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

def load_vgg16(model_dir, gpu_ids):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    # vgg.cuda()
    vgg.cuda(device=gpu_ids[0])
    #加载预训练模型
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    #多卡并行
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg

class SemanticLoss(nn.Module):
    def __init__(self, opt):
        super(SemanticLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(21, affine=False)

    def compute_fcn_loss(self, fcn, img, target):
        img_fcn = vgg_preprocess(img, self.opt)
        target_fcn = vgg_preprocess(target, self.opt)
        img_fea = fcn(img_fcn)
        target_fea = fcn(target_fcn)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
