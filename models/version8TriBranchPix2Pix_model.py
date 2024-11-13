import torch
import torch.nn as nn
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

class version8TriBranchPix2PixModel(BaseModel):
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
            self.model_names = ['G', 'D']
            use_dropout = True
        else:  # during test time, only load G
            self.model_names = ['G']
            use_dropout = False
        # define networks (both generator and discriminator)
        modelG = MultiBranchUNet(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      use_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)

        modelG = init_net(modelG, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG = modelG
        # self.netG = MultiBranchUNet(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = network.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = SSIMLoss(window_size=11, size_average=True)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # opt.lr = 0.0001
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # 创建学习率调度器
            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=50, gamma=0.5)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=50, gamma=0.5)

    def update_learning_rates(self):
        """更新生成器和判别器的学习率"""
        self.scheduler_G.step()
        self.scheduler_D.step()
        print(f'Updated learning rates: G={self.scheduler_G.get_last_lr()[0]}, D={self.scheduler_D.get_last_lr()[0]}')

    def set_input(self, left_images, right_images, mid_images, compound_images):
        # self.target = compound_images
        # self.netG(left_images, right_images, mid_images)
        self.real_L = left_images.to(self.device)
        self.real_A = mid_images.to(self.device)
        self.real_R = right_images.to(self.device)
        self.real_B = compound_images.to(self.device)

    def to(self):
        self.netG = self.netG.to(self.device)

        if self.isTrain:
            self.netD = self.netD.to(self.device)
        return self

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_L, self.real_A, self.real_R)  # G(A)

    def backward_D(self, retain_graph=False):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_graph=retain_graph)

    def backward_G(self, retain_graph=False):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)* 0.01
        # Second, G(A) = B
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_SSIM = self.criterionSSIM(self.fake_B, self.real_B)*1000
        # vgg  = torchvision.models.vgg19(pretrained=True)
        # vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        #
        # feature_extractor = network.FeatureExtractor(vgg)
        self.real_Bf = self.feature_extractor(self.real_B.cpu()).cuda()
        self.fake_Bf = self.feature_extractor(self.fake_B.cpu()).cuda()
        pred_fake1 = torch.cat((self.real_Bf, self.fake_Bf), 1)
        self.contentLoss = self.criterionGAN(pred_fake1, True)
        # self.contentLoss = torch.nn.functional.l1_loss(self.fake_Bf, self.real_Bf)



        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.contentLoss
        # self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.loss_SSIM + self.contentLoss
        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.contentLoss
        self.loss_G.backward(retain_graph=retain_graph)

    def optimize_parameters(self):
        self.forward()  # compute fake image G(A)

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradient to zero
        self.backward_D()  # calculate gradient for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G

        self.optimizer_G.step()  # udpate G's weights

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
        # 创建一个共享的decoder路径
        # self.decoder = Build_decoder(output_nc, ngf, num_downs, norm_layer, use_dropout)
        self.decoder = Build_decoder(input_nc, output_nc, num_downs, ngf=64, norm_layer=norm_layer, use_dropout=False,
                 use_sab=False)
        # 跳跃连接的设置可能需要在UnetSkipConnectionBlock中进行，或者在forward中处理

    # def build_encoder(self, input_nc, ngf, num_downs, norm_layer, use_dropout):
    #     # 构建一个encoder路径
    #     # ...
    #     return encoder
    #
    # def build_decoder(self, output_nc, ngf, num_downs, norm_layer, use_dropout):
    #     # 构建一个decoder路径
    #     # ...
    #     return decoder
    def merge_encoders(self, x1, x2, x3):
        # 拼接三个分支的输出
        # merged = torch.cat([x1, x2, x3], dim=1)  # Concatenate along the channel dimension
        # # 应用池化以减少特征维度
        # # 选择合适的池化层，这里使用自适应平均池化为示例
        # pool = nn.AdaptiveAvgPool2d((x1.size(2), x1.size(3))).to(x1.device)  # size(2) 和 size(3) 是 H 和 W
        # merged = pool(merged)
        # return merged
        # 拼接三个分支的输出
        # 拼接三个分支的输出
        # merged = torch.cat([x1, x2, x3], dim=1)  # Concatenate along the channel dimension
        #
        # merged = self.merge_conv(merged)
        #
        # return merged
        return (x1+x2+x3)/3

    def forward(self, input1, input2, input3):
        # # 下采样
        # encoded1 = self.encoder1(input1)
        # encoded2 = self.encoder2(input2)
        # encoded3 = self.encoder3(input3)
        #
        # # 合并encoder的输出
        # # merged_encoded = torch.cat([encoded1, encoded2, encoded3], dim=1)
        # merged_encoded = self.merge_encoders(encoded1, encoded2, encoded3)
        #
        # # 上采样，并在相应的层添加跳跃连接
        # output = self.decoder(merged_encoded)
        # return output
        encoded1, features1 = self.encoder1(input1)
        encoded2, features2 = self.encoder2(input2)
        encoded3, features3 = self.encoder3(input3)

        # 融合或处理编码输出
        merged_encoded = self.merge_encoders(encoded1, encoded2, encoded3)

        # 解码，并使用跳跃连接特征
        fake_B = self.decoder(merged_encoded, features1, features2, features3)
        return fake_B
        # return output
    def to(self, device):
        # 重写to方法以适应设备变换
        self = super().to(device)  # 首先调用父类的to方法
        # 如果有其他需要特别指定.to(device)的属性，可以在这里添加
        # 例如，如果你有一些缓存的tensors或特殊处理，你可以在这里指定它们
        # self.some_cached_tensor = self.some_cached_tensor.to(device)
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
        # 添加第一个编码层
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.layers.append(EncoderBlock(input_nc, ngf, norm_layer=norm_layer, outermost=True))

        # encoder_block = EncoderBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
        #                                      innermost=True, inter=True)  # add the innermost layer

        # 添加中间编码层
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
        # 返回最终编码器输出和跳连特征列表
        return x, features
        # encoder_block = EncoderBlock(ngf * 4, ngf * 8, input_nc=None, submodule=encoder_block, norm_layer=norm_layer) # 32*24
        # encoder_block = EncoderBlock(ngf * 2, ngf * 4, input_nc=None, submodule=encoder_block, norm_layer=norm_layer) # 64*48
        # encoder_block = EncoderBlock(ngf, ngf * 2, input_nc=None, submodule=encoder_block, norm_layer=norm_layer) # 128*96
        # self.model = EncoderBlock(output_nc, ngf, input_nc=input_nc, submodule=encoder_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer 256*192

    # def forward(self, x):
    #     features = []
    #     for module in self.model.children():
    #         x = module(x)  # 处理当前层
    #         features.append(x)  # 收集当前层的输出作为特征
    #     # 返回最终的编码输出和一个特征列表
    #     # 最终的编码输出是最后一个元素，特征列表用于跳跃连接
    #     return x, features[:-1]

    def to(self, device):
        # 重写to方法以适应设备变换
        self = super().to(device)  # 首先调用父类的to方法
        # 如果有其他需要特别指定.to(device)的属性，可以在这里添加
        # 例如，如果你有一些缓存的tensors或特殊处理，你可以在这里指定它们
        # self.some_cached_tensor = self.some_cached_tensor.to(device)
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
        # 假设features1, features2, features3分别是三个编码器的特征列表
        # 反转每组特征列表，以便于与解码器的层级顺序相匹配
        features1 = list(reversed(features1))
        features2 = list(reversed(features2))
        features3 = list(reversed(features3))

        # 遍历解码器的层级
        # for i, decoder_block in enumerate(self.layers):
        #     if i == 0:  # 假设最外层不需要跳连特征
        #         x = decoder_block(x)
        #     elif i == len(self.layers) - 1:  # 假设最内层不需要额外的跳连特征
        #         x = decoder_block(x)
        #     else:
        #         feature1 = features1[i - 1] if i - 1 < len(features1) else None
        #         feature2 = features2[i - 1] if i - 1 < len(features2) else None
        #         feature3 = features3[i - 1] if i - 1 < len(features3) else None
        #         merged_feature = self.merge_features(feature1, feature2, feature3)
        #         x = decoder_block(x, merged_feature)
        #
        # return x
        # 遍历解码器的层级
        for i, decoder_block in enumerate(self.layers):
            # 对于每一层，从每组特征中取出相应的特征
            # 注意，这里假设所有编码器和解码器的层数相同，实际使用时可能需要调整
            feature1 = features1[i] if i < len(features1) else None
            feature2 = features2[i] if i < len(features2) else None
            feature3 = features3[i] if i < len(features3) else None

            # 将来自三个编码器的特征进行融合，具体的融合方法根据任务需求设计
            # 这里只是一个简单的示例，实际应用中可能需要更复杂的融合策略
            # merged_feature = self.merge_features(feature1, feature2, feature3)

            # 将融合后的特征与当前解码器的输入x进行合并
            # 然后通过当前的解码器块进行处理
            x = decoder_block(x, feature1, feature2, feature3)

        return x

    # def merge_features(self, f1, f2, f3):
    #     # 根据需要选择融合策略，例如简单的加和或者更复杂的融合方法
    #     return (f1 + f2 + f3)/3
    #     # return torch.cat([f1, f2, f3], dim=1)
    def merge_features(self, f1, f2, f3):
        # 拼接特征
        merged = torch.cat([f1, f2, f3], dim=1)
        # # 应用一个卷积层来减少维度
        # conv = nn.Conv2d(f1.size(1) * 3, f1.size(1), kernel_size=1, padding=0).to(f1.device)  # 假设f1, f2, f3具有相同的通道数
        # merged = conv(merged)
        return merged

    def to(self, device):
        # 重写to方法以适应设备变换
        self = super().to(device)  # 首先调用父类的to方法
        # 如果有其他需要特别指定.to(device)的属性，可以在这里添加
        # 例如，如果你有一些缓存的tensors或特殊处理，你可以在这里指定它们
        # self.some_cached_tensor = self.some_cached_tensor.to(device)
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
        # 重写to方法以适应设备变换
        self = super().to(device)  # 首先调用父类的to方法
        # 如果有其他需要特别指定.to(device)的属性，可以在这里添加
        # 例如，如果你有一些缓存的tensors或特殊处理，你可以在这里指定它们
        # self.some_cached_tensor = self.some_cached_tensor.to(device)
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

        # 对于非最内层的解码块，我们预计会有跳连特征，因此输入通道数是 inner_nc + skip_channels
        # upsample_input_nc = inner_nc if innermost else (inner_nc + skip_channels)
        #
        # # 对于最外层，无需跳连特征融合
        # output_nc = outer_nc if not outermost else input_nc

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
            # 对于最外层，直接应用模型（没有跳连特征需要融合） 试试看跳连
            ## skip_features_enchance = self.sge(skip_features)
            # skip_features_enchance = torch.mul(self.sge(skip_features), skip_features)
            # x = torch.cat([x, skip_features_enchance], dim=1) if skip_features is not None else x
            # skip_features = self.featureskip(feature1, feature2, feature3)
            skip_features = (feature1 + feature2 + feature3) / 3

            x = torch.cat([x, skip_features], dim=1) if skip_features is not None else x
            self.pixelatt(x)
            x = self.model(x)
        elif self.innermost:
            # 对于最内层，没有跳连特征，直接通过模型
            x = self.model(x)
        else:
            # 对于中间层，融合跳连特征
            ## skip_features_enchance = self.sge(skip_features)
            # skip_features_enchance = torch.mul(self.sge(skip_features), skip_features)
            # x = torch.cat([x, skip_features_enchance], dim=1) if skip_features is not None else x
            skip_features = self.featureskip(feature1, feature2, feature3)

            x = torch.cat([x, skip_features], dim=1) if skip_features is not None else x
            self.pixelatt(x)
            x = self.model(x)
            # 可选：应用PixelAwareAttention或其他注意力机制

        return x
        # # 对于最内层，直接通过模型处理
        # if self.innermost:
        #     return self.model(x)
        # # 对于中间层和最外层，如果有跳连特征，则融合这些特征
        # elif skip_features is not None:
        #     x = torch.cat([x, skip_features], dim=1)
        # return self.model(x)
        # if self.innermost:
        #     # 最内层：直接应用PixelAwareAttention
        #     return self.pixelatt(self.model(x))
        # else:
        #     # 对于中间层和最外层
        #     if skip_features is not None:
        #         x = torch.cat([x, skip_features], dim=1)
        #     x = self.model(x)  # 首先递归处理子模块
        #
        #     # 然后应用PixelAwareAttention
        #     x = self.pixelatt(x)
        #     return x

    def to(self, device):
        # 重写to方法以适应设备变换
        self = super().to(device)  # 首先调用父类的to方法
        # 如果有其他需要特别指定.to(device)的属性，可以在这里添加
        # 例如，如果你有一些缓存的tensors或特殊处理，你可以在这里指定它们
        # self.some_cached_tensor = self.some_cached_tensor.to(device)
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

# class MultiBranchFeatureFusion(nn.Module):
#     def __init__(self, channel):
#         super(MultiBranchFeatureFusion, self).__init__()
#         self.spatial_att = SpatialAttention()
#         self.channel_att = ChannelAttention(channel * 3)
#         self.conv_reduce = nn.Conv2d(channel * 3, channel, kernel_size=1)
#
#     def forward(self, x1, x2, x3):
#         # 空间注意力
#         spatial_mask_x1 = self.spatial_att(x1)
#         x1 = x1 * spatial_mask_x1
#
#         spatial_mask_x2 = self.spatial_att(x2)
#         x2 = x2 * spatial_mask_x2
#
#         spatial_mask_x3 = self.spatial_att(x3)
#         x3 = x3 * spatial_mask_x3
#         # 合并特征
#         x = torch.cat([x1, x2, x3], dim=1)
#
#         # 通道注意力
#         x = self.channel_att(x)
#         # 通道减少
#         x = self.conv_reduce(x)
#         return x
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
        attention_ba = self.attentionAB(a, b)  # A对B的影响
        attention_bc = self.attentionBC(c, b)  # C对B的影响

        # # 融合注意力权重
        # combined_attention = (attention_ba + attention_bc) / 2
        #
        # # 应用融合注意力到B的值
        # values = self.value_conv(b).view(b.size(0), -1, b.size(-2)*b.size(-1))  # [n, C, H*W]
        values_b = self.value_conv(b)  # 假设已经定义了适当的卷积层
        values_b = values_b.view(b.size(0), -1, b.size(-2) * b.size(-1))  # 调整形状以适应矩阵乘法
        # 应用注意力分数
        enhanced_b_ba = torch.bmm(values_b, attention_ba.transpose(1, 2))
        enhanced_b_bc = torch.bmm(values_b, attention_bc.transpose(1, 2))

        # 融合两种影响
        enhanced_b = (enhanced_b_ba + enhanced_b_bc) / 2
        # enhanced_b = torch.bmm(values, combined_attention.transpose(1, 2))  # [n, C, H*W]
        enhanced_b = enhanced_b.view_as(b)  # Reshape to the original b shape

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

# SSIM损失的简单实现
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1

    def forward(self, img1, img2):
        # 计算SSIM分数
        ssim_value = ssim(img1, img2, win_size=self.window_size, size_average=self.size_average)
        # 使用1 - SSIM分数来计算损失
        return 1 - ssim_value