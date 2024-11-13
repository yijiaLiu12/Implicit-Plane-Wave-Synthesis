from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# import torchvision
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import copy
from models.SAB import ChannelAttention,SpatialAttention,SpatioAttention, LocalAwareAttention, GlobalAwareAttention, PixelAwareAttention

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance': # InstanceNormalization
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier': # 
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):# 定义神经网络（G和D）
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_sab=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
    elif netG == 'srgan':
        net = SRGenerator()
    elif netG == 'VDSR':
        net = VDCGAN()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'srgan':
        net = SRDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        # elif gan_mode == 'style_texture':
            # self.loss = get_style_texture_algorithm()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp': #WGAN损失函数为什么时求均值？
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 3
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # print(input.shape)
        """Standard forward"""
        x1 = self.model(input)
        return x1

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.la = LocalAwareAttention()

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        # print(x.shape)
        x1 = self.conv_block(x)
        out = x + x1  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_sab=False):
        """Construct a Unet generator
        Parameters:

            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, inter=True)  # add the innermost layer
        # model = [unet_block, LocalAwareAttention()]
        # unet_block = nn.Sequential(*model)
        # unet_block = UnetSkipConnectionBlock(ngf * 2 , ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer,innermost=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # model = [ResnetBlock(ngf * 8, padding_type="reflect", norm_layer=norm_layer, use_dropout=use_dropout,
        #                        use_bias=use_bias)]
        # model = []

        # if use_sab:
        #     unet_block = SpatialAttention(ngf*8)
        # for i in range(num_down - 5):

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, inter=True)

            # model = [unet_block, LocalAwareAttention()]
            # unet_block = nn.Sequential(*model)

        #     model += [ResnetBlock(ngf * 8, padding_type="reflect", norm_layer=norm_layer, use_dropout=use_dropout,
        #                           use_bias=use_bias)]
        # unet_block = nn.Sequential(*model)
        # # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
        #                                      norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer) # 32*24
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer) # 64*48
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer) # 128*96
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer 256*192

        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
        #                                      norm_layer=norm_layer, s1=32, s2=24, use_gb=True)  # 32*24
        # unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
        #                                      norm_layer=norm_layer, s1=64, s2=48,use_gb=True)  # 64*48
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
        #                                      s1=128, s2=96,use_gb=True)  # 128*96
        # self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
        #                                      norm_layer=norm_layer,use_gb=False)  # add the outermost layer 256*192

    def forward(self, input):

        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):

    """Defines the Unet submodule with skip connection.
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, inter=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.inter = inter

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        self.input_nc = input_nc
        self.outer_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, False)
        upnorm = norm_layer(outer_nc)
        # localatt = LocalAwareAttention()
        self.pixelatt = PixelAwareAttention(inner_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            # up = [upconv, uprelu, nn.Tanh()]
            up = [upconv, uprelu]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu]
            if self.inter == True:
                up = [upconv, uprelu, upnorm, localatt]

            else:
                up = [upconv, uprelu, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu, downnorm]
            if self.inter == True:
                up = [upconv, uprelu, upnorm, localatt]

            else:
                up = [upconv, uprelu, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        # self.model1 = nn.Sequential(*model1)
        # self.sa = SpatialAttention()
        # self.ca = ChannelAttention(input_nc)
        self.pa = PixelAwareAttention(input_nc)
        # self.la = LocalAwareAttention()



    def forward(self, x):
        if self.outermost:
            return self.model(x)

           # add skip connections
        # else:
        #     return torch.cat([x, self.model(x)], 1)
        else:
            if self.inter == False:
                x2 = self.pa(x)
                x3 = torch.mul(x2, x)
                # print(x.shape)
                return torch.cat([x3, self.model(x)], 1)
            else:
                return torch.cat([x, self.model(x)], 1)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock,self).__init__()

        self.model = [nn.Conv2d(64, 64, kernel_size=(5,3), padding=(2,1)),
                      nn.BatchNorm2d(64),
                      nn.ReLU(),
                      nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2, 1)),
                      nn.BatchNorm2d(64)
                      ]
        self.model = nn.Sequential(*self.model)
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=(5,3), padding=(2,1))
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2, 1))
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()

    def forward(self, x):
        return self.model(x) + x



class SRGenerator(nn.Module):
    """Defines a super resolution GAN generator"""
    def __init__(self):
        super(SRGenerator, self).__init__()

        self.model = [nn.Conv2d(1, 64, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2, 1)),
                      nn.BatchNorm2d(64),
                      nn.ReLU(),
                      ResidualBlock(),
                      ResidualBlock(),
                      ResidualBlock(),
                      ResidualBlock(),
                      ResidualBlock(),
                      ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      # ResidualBlock(),
                      nn.Conv2d(64, 64, kernel_size=3, padding=1),
                      nn.BatchNorm2d(64),
                      ]
        self.model = nn.Sequential(*self.model)

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 3), padding=(2, 1))
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2, 1))
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)

        # self.res_block = [ResidualBlock()] * 5
        # self.res_block = nn.Sequential(*self.res_block)
        # print(self.res_block)
        # self.res_block1 = ResidualBlock()
        # self.res_block2 = ResidualBlock()
        # self.res_block3 = ResidualBlock()
        # self.res_block4 = ResidualBlock()
        # self.res_block5 = ResidualBlock()


    def forward(self, x):
        # print(x.shape)
        # print(self.conv1)
        # x1 = self.conv1(x)
        # x1 = self.relu(x1)
        #
        # x2 = self.res_block1(x1)
        # x3 = self.res_block2(x2)
        # x4 = self.res_block3(x3)
        # x5 = self.res_block4(x4)
        # x6 = self.res_block5(x5)
        # # x2 = self.res_block(x)
        # x7 = self.conv2(x6)
        # x7 = self.bn(x7)
        x1 = self.model(x)
        x2 = x1 + x
        x2 = self.conv3(x2)

        return x2

class SRDiscriminator(nn.Module):
    """Defines a super resolution GAN discriminator"""
    def __init__(self):
        super(SRDiscriminator, self).__init__()

        # self.net = [nn.Conv2d(1, 64, kernel_size=3, stride=1),
        #             nn.LeakyReLU(),
        #             self.conv_block(64, 64, 2),
        #             self.conv_block(64, 128, 1),
        #             self.conv_block(128, 128, 2),
        #             self.conv_block(128, 256, 1),
        #             self.conv_block(256, 256, 2),
        #             self.conv_block(256, 512, 1),
        #             self.conv_block(512, 512, 2),
        #             nn.Conv2d(512, 1024, kernel_size=1),
        #             nn.LeakyReLU(),
        #             nn.Conv2d(1024, 1, kernel_size=1)
        #             ]
        # self.net = nn.Sequential(*self.net)

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1)

        self.con_block1 = self.conv_block(64, 64, 2)
        self.con_block2 = self.conv_block(64, 128, 1)
        self.con_block3 = self.conv_block(128, 128, 2)
        self.con_block4 = self.conv_block(128, 256, 1)
        self.con_block5 = self.conv_block(256, 256, 2)
        self.con_block6 = self.conv_block(256, 512, 1)
        self.con_block7 = self.conv_block(512, 512, 2)

        self.dense1 = nn.Conv2d(512, 1024, kernel_size=1)
        self.dense2 = nn.Conv2d(1024, 1, kernel_size=1)

        self.Lrelu = nn.LeakyReLU()

    def conv_block(self, in_num, out_num, stride):
        con_block = []
        con_block += [nn.Conv2d(in_num, out_num, kernel_size=3, stride=stride)]
        con_block += [nn.BatchNorm2d(out_num), nn.LeakyReLU()]

        return nn.Sequential(*con_block)

    def forward(self, x):
        # print(x.shape)
        x1 = self.conv1(x)
        x1 = self.Lrelu(x1)

        x2 = self.con_block1(x1)
        x3 = self.con_block2(x2)
        x4 = self.con_block3(x3)
        x5 = self.con_block4(x4)
        x6 = self.con_block5(x5)
        x7 = self.con_block6(x6)
        x8 = self.con_block7(x7)

        x9 = self.dense1(x8)
        x9 = self.Lrelu(x9)
        x10 = self.dense2(x9)

        return x10

class VDCGAN(nn.Module):
    def __init__(self):
        """Defines a very deep convolutional nerual network"""
        super(VDCGAN, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)

        # Residual layer
        conv_block1 = []
        for i in range(3):
            j1 = 64* (2 **i)
            j2 = 64* (2 **(i+1))
            conv_block1.append(nn.Conv2d(j1, j2, kernel_size=4, stride=2, padding=1, bias=False))
            conv_block1.append(nn.ReLU(inplace=True))
        self.conv_block1 = nn.Sequential(*conv_block1)

        conv_block2 = []
        for i in range(3):
            j1 = 64 * (2 **(3-i))
            j2 = 64 * (2 **(3-i-1))
            conv_block2.append(nn.ConvTranspose2d(j1, j2, kernel_size=4, stride=2, padding=1, bias=False))
            conv_block2.append(nn.ReLU(inplace=True))
        self.conv_block2 = nn.Sequential(*conv_block2)


        # last layer
        self.conv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv_block1(out)
        out = self.conv_block2(out)
        out = self.conv2(out)
        return out







