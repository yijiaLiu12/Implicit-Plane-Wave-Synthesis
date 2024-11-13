import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import network
import torchvision
import functools
from model.attention.SGE import SpatialGroupEnhance
# from thop import profile
from models.SAB import CrissCrossAttention
from models.SAB import ChannelAttention,SpatialAttention,SpatioAttention, LocalAwareAttention, GlobalAwareAttention, PixelAwareAttention
import os
from torch.nn import DataParallel
from torch.nn import init
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class TriBranchCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper
        parser.set_defaults(norm='instance', netG='unet_128', netD='basic', use_sab=False, name='unet_b002')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1000, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)

        self.feature_extractor = network.FeatureExtractor(vgg)

        if self.isTrain:
            # self.model_names = ['G', 'D']
            self.model_names = ['G_Stage1_Left','G_Stage1_Right', 'G_Stage2', 'D_Stage1_Left', 'D_Stage1_Right', 'D_Stage2']
            use_dropout = True
        else:  # during test time, only load G
            # self.model_names = ['G']
            self.model_names = ['G_Stage1_Left','G_Stage1_Right', 'G_Stage2']
            use_dropout = False

            # define networks (both generator and discriminator)
        self.netG_Stage1_Left = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)
        self.netG_Stage1_Right = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                                 opt.use_sab)
        # define networks (both generator and discriminator)
        modelG = MultiBranchUNet(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      use_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)
        modelG = init_net(modelG, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG = modelG
        self.netG_Stage2 = modelG
        # self.netG = MultiBranchUNet(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD_Stage1_Left = network.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_Stage1_Right = network.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_Stage2 = network.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = SSIMLoss(window_size=11, size_average=True)

            self.optimizer_G1 = torch.optim.Adam(
                itertools.chain(self.netG_Stage1_Left.parameters(), self.netG_Stage1_Right.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizer_G2 = torch.optim.Adam(
                self.netG_Stage2.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizer_D1 = torch.optim.Adam(
                itertools.chain(self.netD_Stage1_Left.parameters(), self.netD_Stage1_Right.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizer_D2 = torch.optim.Adam(
                self.netD_Stage2.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)

            # 创建学习率调度器
            # self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=50, gamma=0.5)
            # self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=50, gamma=0.5)
            self.scheduler_G1 = torch.optim.lr_scheduler.StepLR(self.optimizer_G1, step_size=10, gamma=0.95)
            self.scheduler_G2 = torch.optim.lr_scheduler.StepLR(self.optimizer_G2, step_size=10, gamma=0.95)
            self.scheduler_D1 = torch.optim.lr_scheduler.StepLR(self.optimizer_D1, step_size=10, gamma=0.95)
            self.scheduler_D2 = torch.optim.lr_scheduler.StepLR(self.optimizer_D2, step_size=10, gamma=0.95)

    def update_learning_rates(self):
        """更新生成器和判别器的学习率"""
        self.scheduler_G1.step()
        self.scheduler_G2.step()
        self.scheduler_D1.step()
        self.scheduler_D2.step()
        print(f'Updated learning rates: G1={self.scheduler_G1.get_last_lr()[0]}, D1={self.scheduler_D1.get_last_lr()[0]},G2={self.scheduler_G2.get_last_lr()[0]}, D2={self.scheduler_D2.get_last_lr()[0]}')

    def set_input(self, left_images, right_images, mid_images, compound_images):
        # self.target = compound_images
        # self.netG(left_images, right_images, mid_images)
        self.real_L = left_images.to(self.device)
        self.real_mid = mid_images.to(self.device)
        self.real_R = right_images.to(self.device)
        self.real_Compound = compound_images.to(self.device)

    def to(self):
        # self.netG = self.netG.to(self.device)
        self.netG_Stage1_Left = self.netG_Stage1_Left.to(self.device)
        self.netG_Stage1_Right = self.netG_Stage1_Right.to(self.device)
        self.netG_Stage2 = self.netG_Stage2.to(self.device)
        if self.isTrain:
            self.netD_Stage1_Left = self.netD_Stage1_Left.to(self.device)
            self.netD_Stage1_Right = self.netD_Stage1_Right.to(self.device)
            self.netD_Stage2 = self.netD_Stage2.to(self.device)
        return self

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_L = self.netG_Stage1_Left(self.real_mid)
        self.fake_R = self.netG_Stage1_Right(self.real_mid)
        self.fake_Compound = self.netG_Stage2(self.fake_L, self.real_mid, self.fake_R)  # fake compound

    def backward_D1_Left(self, retain_graph=False):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_ML = torch.cat((self.real_mid, self.fake_L),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD_Stage1_Left(fake_ML.detach())
        self.loss_D_fake_L = self.criterionGAN(pred_fake, False)

        # Real
        real_ML = torch.cat((self.real_mid, self.real_L), 1)
        pred_real = self.netD_Stage1_Left(real_ML)
        self.loss_D_real_L = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D1_L = (self.loss_D_fake_L + self.loss_D_real_L) * 0.5
        self.loss_D1_L.backward(retain_graph=retain_graph)

    def backward_D1_Right(self, retain_graph=False):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_MR = torch.cat((self.real_mid, self.fake_R),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD_Stage1_Right(fake_MR.detach())
        self.loss_D_fake_R = self.criterionGAN(pred_fake, False)

        # Real
        real_MR = torch.cat((self.real_mid, self.real_R), 1)
        pred_real = self.netD_Stage1_Right(real_MR)
        self.loss_D_real_R = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D1_R = (self.loss_D_fake_R + self.loss_D_real_R) * 0.5
        self.loss_D1_R.backward(retain_graph=retain_graph)

    def backward_D2(self, retain_graph=False):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_MC = torch.cat((self.real_mid, self.fake_Compound),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD_Stage2(fake_MC.detach())
        self.loss_D_fake_Compound = self.criterionGAN(pred_fake, False)

        # Real
        real_MC = torch.cat((self.real_mid, self.real_Compound), 1)
        pred_real = self.netD_Stage1_Right(real_MC)
        self.loss_D_real_Compound = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D2 = (self.loss_D_fake_Compound + self.loss_D_real_Compound) * 0.5
        self.loss_D2.backward(retain_graph=retain_graph)

    def backward_G1_L(self, retain_graph=False):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_ML = torch.cat((self.real_mid, self.fake_L), 1)
        pred_fake = self.netD_Stage1_Left(fake_ML)
        self.loss_G1_L_GAN = self.criterionGAN(pred_fake, True)* 0.01
        # Second, G(A) = B
        self.loss_G1_L_L2 = self.criterionL2(self.fake_L, self.real_L) * self.opt.lambda_L1
        # self.loss_SSIM = self.criterionSSIM(self.fake_B, self.real_B)*1000
        # vgg  = torchvision.models.vgg19(pretrained=True)
        # vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        #
        # feature_extractor = network.FeatureExtractor(vgg)
        self.real_Lf = self.feature_extractor(self.real_L.cpu()).cuda()
        self.fake_Lf = self.feature_extractor(self.fake_L.cpu()).cuda()
        pred_fake1 = torch.cat((self.real_Lf, self.fake_Lf), 1)
        self.L_contentLoss = self.criterionGAN(pred_fake1, True)
        # self.contentLoss = torch.nn.functional.l1_loss(self.fake_Bf, self.real_Bf)



        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.contentLoss
        # self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.loss_SSIM + self.contentLoss
        self.loss_G1_L = self.loss_G1_L_GAN + self.loss_G1_L_L2 + self.L_contentLoss
        self.loss_G1_L.backward(retain_graph=retain_graph)

    def backward_G1_R(self, retain_graph=False):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_MR = torch.cat((self.real_mid, self.fake_R), 1)
        pred_fake = self.netD_Stage1_Right(fake_MR)
        self.loss_G1_R_GAN = self.criterionGAN(pred_fake, True) * 0.01
        # Second, G(A) = B
        self.loss_G1_R_L2 = self.criterionL2(self.fake_R, self.real_R) * self.opt.lambda_L1

        self.real_Rf = self.feature_extractor(self.real_R.cpu()).cuda()
        self.fake_Rf = self.feature_extractor(self.fake_R.cpu()).cuda()
        pred_fake1 = torch.cat((self.real_Rf, self.fake_Rf), 1)
        self.R_contentLoss = self.criterionGAN(pred_fake1, True)
        # self.contentLoss = torch.nn.functional.l1_loss(self.fake_Bf, self.real_Bf)


        self.loss_G1_R = self.loss_G1_R_GAN + self.loss_G1_R_L2 + self.R_contentLoss
        self.loss_G1_R.backward(retain_graph=retain_graph)
        
    def backward_G2(self, retain_graph=False):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_MC = torch.cat((self.real_mid, self.fake_Compound), 1)
        pred_fake = self.netD_Stage2(fake_MC)
        self.loss_G2_GAN = self.criterionGAN(pred_fake, True) * 0.01
        # Second, G(A) = B
        self.loss_G2_L2 = self.criterionL2(self.fake_Compound, self.real_Compound) * self.opt.lambda_L1
        # self.loss_SSIM = self.criterionSSIM(self.fake_B, self.real_B)*1000
        # vgg  = torchvision.models.vgg19(pretrained=True)
        # vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        #
        # feature_extractor = network.FeatureExtractor(vgg)
        self.real_Cf = self.feature_extractor(self.real_Compound.cpu()).cuda()
        self.fake_Cf = self.feature_extractor(self.fake_Compound.cpu()).cuda()
        pred_fake1 = torch.cat((self.real_Cf, self.fake_Cf), 1)
        self.C_contentLoss = self.criterionGAN(pred_fake1, True)
        # self.contentLoss = torch.nn.functional.l1_loss(self.fake_Bf, self.real_Bf)


        self.loss_G2 = self.loss_G2_GAN + self.loss_G2_L2 + self.C_contentLoss
        self.loss_G2.backward(retain_graph=retain_graph)

    def optimize_parameters(self):
        self.forward()  # 计算前向传播

        # 更新第二阶段的判别器
        self.set_requires_grad(self.netD_Stage2, True)  # 为第二阶段的判别器启用梯度计算
        self.optimizer_D2.zero_grad()  # 清空第二阶段判别器的梯度
        self.backward_D2(retain_graph=True)  # 计算第二阶段判别器的梯度
        self.optimizer_D2.step()  # 更新第二阶段判别器的权重

        # 更新第二阶段生成器
        self.set_requires_grad(self.netD_Stage2, False)  # 在优化生成器时，第二阶段的判别器不需要梯度
        self.optimizer_G2.zero_grad()  # 清空第二阶段生成器的梯度
        self.backward_G2(retain_graph=True)  # 计算第二阶段生成器的梯度
        self.optimizer_G2.step()  # 更新第二阶段生成器的权重

        # 更新第一阶段的判别器（左和右）
        self.set_requires_grad([self.netD_Stage1_Left, self.netD_Stage1_Right], True)  # 为第一阶段的判别器启用梯度计算
        self.optimizer_D1.zero_grad()  # 清空第一阶段判别器的梯度
        self.backward_D1_Left(retain_graph=True)  # 计算左侧判别器的梯度，保留计算图
        self.backward_D1_Right()  # 计算右侧判别器的梯度
        self.optimizer_D1.step()  # 更新第一阶段判别器的权重

        # 更新生成器（第一阶段左和右）
        self.set_requires_grad([self.netD_Stage1_Left, self.netD_Stage1_Right], False)  # 在优化生成器时，第一阶段的判别器不需要梯度
        self.optimizer_G1.zero_grad()  # 清空第一阶段生成器的梯度
        self.backward_G1_L(retain_graph=True)  # 计算左侧生成器的梯度，保留计算图
        self.backward_G1_R()  # 计算右侧生成器的梯度
        self.optimizer_G1.step()  # 更新第一阶段生成器的权重

    # def save_networks(self, epoch, networkname=None):
    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

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


class MultiBranchUNet(nn.Module):

    # def __init__(self, input_nc, output_nc, ngf, num_downs, norm_layer, use_dropout):
    def __init__(self, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_sab=False):
        super(MultiBranchUNet, self).__init__()
        if netG == 'unet_128':
            num_downs = 7
        norm_layer = get_norm_layer(norm_type=norm)
        # 创建三个独立的encoder路径
        self.encoder1 = Build_encoder(input_nc, output_nc, num_downs, ngf=64, norm_layer=norm_layer, use_dropout=use_dropout,
                 use_sab=False)
        self.encoder2 = Build_encoder(input_nc, output_nc, num_downs, ngf=64, norm_layer=norm_layer, use_dropout=use_dropout,
                 use_sab=False)
        self.encoder3 = Build_encoder(input_nc, output_nc, num_downs, ngf=64, norm_layer=norm_layer, use_dropout=use_dropout,
                 use_sab=False)

        self.merge_conv = nn.Conv2d(ngf * 8 * 3, ngf * 8, kernel_size=1, bias=False)#底层特征融合卷积
        self.decoder = Build_decoder(input_nc, output_nc, num_downs, ngf=64, norm_layer=norm_layer, use_dropout=False,
                 use_sab=False)



    def merge_encoders(self, x1, x2, x3):

        return (x1+x2+x3)/3

    def forward(self, input1, input2, input3):

        encoded1, features1 = self.encoder1(input1)
        encoded2, features2 = self.encoder2(input2)
        encoded3, features3 = self.encoder3(input3)

        # 融合或处理编码输出
        merged_encoded = self.merge_encoders(encoded1, encoded2, encoded3)

        fake_B = self.decoder(merged_encoded, features1, features2, features3)
        return fake_B
        # return output
    def to(self, device):
        self = super().to(device)
        return self





class Build_encoder(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_sab=False):
        """Construct a downsample encoder
        Parameters:

            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the downsample encoder from the innermost layer to the outermost layer.
        """
        super(Build_encoder, self).__init__()
        self.layers = nn.ModuleList()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.layers.append(EncoderBlock(input_nc, ngf, norm_layer=norm_layer, outermost=True))

        mult = 1
        for i in range(3):
            self.layers.append(EncoderBlock(ngf * mult, ngf * mult * 2, norm_layer=norm_layer, use_dropout=use_dropout))
            mult *= 2

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            self.layers.append(EncoderBlock(ngf * 8, ngf * 8, input_nc=None,  norm_layer=norm_layer, use_dropout=use_dropout, inter=True))
            # encoder_block = EncoderBlock(ngf * 8, ngf * 8, input_nc=None, submodule=encoder_block, norm_layer=norm_layer, use_dropout=use_dropout, inter=True)

        self.layers.append(EncoderBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, inter=True))  # add the innermost layer

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return x, features

    def to(self, device):
        self = super().to(device)
        return self

class Build_decoder(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_sab=False):
        """Construct a downsample encoder
        Parameters:

            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the downsample encoder from the innermost layer to the outermost layer.
        """
        super(Build_decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(DecoderBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, inter=True)) # add the innermost layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            self.layers.append(DecoderBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout, inter=True))

        self.layers.append(DecoderBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)) # 32*24
        self.layers.append(DecoderBlock(ngf * 2, ngf * 4, input_nc=None,  norm_layer=norm_layer)) # 64*48
        self.layers.append(DecoderBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)) # 128*96
        self.layers.append(DecoderBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)) # add the outermost layer 256*192

    def forward(self, x, features1, features2, features3):
        features1 = list(reversed(features1))
        features2 = list(reversed(features2))
        features3 = list(reversed(features3))

        for i, decoder_block in enumerate(self.layers):
            feature1 = features1[i] if i < len(features1) else None
            feature2 = features2[i] if i < len(features2) else None
            feature3 = features3[i] if i < len(features3) else None

            x = decoder_block(x, feature1, feature2, feature3)

        return x

    def merge_features(self, f1, f2, f3):
        # 拼接特征
        merged = torch.cat([f1, f2, f3], dim=1)

        return merged

    def to(self, device):
        self = super().to(device)
        return self


class EncoderBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 inter=False):
        """Construct a Unet encoder submodule .

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
        super(EncoderBlock, self).__init__()
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

        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu]
            model = down
        else:
            down = [downconv, downrelu, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down

        self.model = nn.Sequential(*model)


    def forward(self, input):

        """Standard forward"""
        return self.model(input)

    def to(self, device):
        self = super().to(device)
        return self

class DecoderBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, skip_channels=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 inter=False):
        """Construct a Unet decoder submodule.

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
        super(DecoderBlock, self).__init__()
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


        uprelu = nn.LeakyReLU(0.2, False)
        upnorm = norm_layer(outer_nc)
        #localatt = LocalAwareAttention()
        CCatt = CrissCrossAttention(inner_nc)
        self.pixelatt = PixelAwareAttention(inner_nc* 2)

        if outermost:
            self.pixelatt = PixelAwareAttention(inner_nc*2)
            self.featureskip = MultiBranchAttention(inner_nc)
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            up = [upconv, uprelu]
            model =  up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=1,
                                        padding=1, bias=use_bias)
            if self.inter == True:
                #up = [upconv, uprelu, upnorm, localatt]
                up = [upconv, uprelu, upnorm, CCatt]

            else:
                up = [upconv, uprelu, upnorm]
            model = up
        else:
            self.featureskip = MultiBranchAttention(inner_nc)
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if self.inter == True:
                #up = [upconv, uprelu, upnorm, localatt]
                up = [upconv, uprelu, upnorm, CCatt]

            else:
                up = [upconv, uprelu, upnorm]

            if use_dropout:
                model = up + [nn.Dropout(0.5)]
            else:
                model = up

        self.model = nn.Sequential(*model)
        self.sge = SpatialGroupEnhance(groups=8)

    def forward(self, x, feature1=None, feature2=None, feature3=None):
        if self.outermost:
            skip_features = (feature1 + feature2 + feature3) / 3

            x = torch.cat([x, skip_features], dim=1) if skip_features is not None else x
            self.pixelatt(x)
            x = self.model(x)
        elif self.innermost:

            x = self.model(x)
        else:
            skip_features = self.featureskip(feature1, feature2, feature3)

            x = torch.cat([x, skip_features], dim=1) if skip_features is not None else x
            self.pixelatt(x)
            x = self.model(x)


        return x


    def to(self, device):

        self = super().to(device)
        return self


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiBranchFeatureFusion(nn.Module):
    def __init__(self, channel):
        super(MultiBranchFeatureFusion, self).__init__()
        self.spatial_att = SpatialAttention()
        self.channel_att = ChannelAttention(channel * 3)
        self.conv_reduce = nn.Conv2d(channel * 3, channel, kernel_size=1)

    def forward(self, x1, x2, x3):
        # 空间注意力
        spatial_mask_x1 = self.spatial_att(x1)
        x1 = x1 * spatial_mask_x1

        spatial_mask_x2 = self.spatial_att(x2)
        x2 = x2 * spatial_mask_x2

        spatial_mask_x3 = self.spatial_att(x3)
        x3 = x3 * spatial_mask_x3
        # 合并特征
        x = torch.cat([x1, x2, x3], dim=1)

        # 通道注意力
        x = self.channel_att(x)
        # 通道减少
        x = self.conv_reduce(x)
        return x
class AttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        m, n = x1.size(0), x2.size(0)
        queries = self.query_conv(x1).view(m, -1, x1.size(-2)*x1.size(-1))  # [m, C', H*W]
        keys = self.key_conv(x2).view(n, -1, x2.size(-2)*x2.size(-1))       # [n, C', H*W]

        attention = torch.bmm(queries.transpose(1, 2), keys)  # [m, H*W, H*W]
        attention = self.softmax(attention)  # Softmax over the last dimension
        return attention

class MultiBranchAttention(nn.Module):
    def __init__(self, channels):
        super(MultiBranchAttention, self).__init__()
        self.attentionAB = AttentionModule(channels)
        self.attentionBC = AttentionModule(channels)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, a, b, c):
        attention_ba = self.attentionAB(a, b)
        attention_bc = self.attentionBC(c, b)

        values_b = self.value_conv(b)
        values_b = values_b.view(b.size(0), -1, b.size(-2) * b.size(-1))
        # 应用注意力分数
        enhanced_b_ba = torch.bmm(values_b, attention_ba.transpose(1, 2))
        enhanced_b_bc = torch.bmm(values_b, attention_bc.transpose(1, 2))

        enhanced_b = (enhanced_b_ba + enhanced_b_bc) / 2
        enhanced_b = enhanced_b.view_as(b)

        return enhanced_b
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


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1

    def forward(self, img1, img2):
        # 计算SSIM分数
        ssim_value = ssim(img1, img2, win_size=self.window_size, size_average=self.size_average)

        return 1 - ssim_value