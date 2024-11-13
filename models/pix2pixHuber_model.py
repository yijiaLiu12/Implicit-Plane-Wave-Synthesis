import torch
from .base_model import BaseModel
from . import network
import torchvision
#from thop import profile


class pix2pixHuberModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    """

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
        parser.set_defaults(norm='instance', netG='unet_128', netD='basic', use_sab=False, name='unet_whole001')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        vgg = torchvision.models.vgg19(pretrained=True).to(self.device)
        vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)

        self.feature_extractor = network.FeatureExtractor(vgg)
        self.feature_extractor.to(self.device)
        # # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = network.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionHuber = torch.nn.SmoothL1Loss(beta=1.0)  # 可以调整beta的值来设定delta
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def l1_regularization(self, weight, l1_factor):
        l1_reg = torch.norm(weight, p=1)
        return l1_factor * l1_reg

    def set_input(self, data, target):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input : low resolution image
            target : high resulotion image
        """
        self.real_A = data.to(self.device)
        self.real_B = target.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        # self.total_ops, self.total_params = profile(self.netG.cuda(), (self.real_A.cuda(),))
        # print(self.total_ops, self.total_params)



    def backward_D(self,retain_graph=False):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) *0.5
        self.loss_D.backward(retain_graph=retain_graph)

    def backward_G(self,retain_graph=False):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Replace L2 loss with Huber loss
        self.loss_G_Huber = self.criterionHuber(self.fake_B, self.real_B) * self.opt.lambda_L1
        # vgg  = torchvision.models.vgg19(pretrained=True)
        # vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # L1 regularization on generator weights
        l1_reg = sum(self.l1_regularization(param, 1e-5) for param in self.netG.parameters())

        # feature_extractor = network.FeatureExtractor(vgg)
        self.real_Bf = self.feature_extractor(self.real_B).cuda()
        self.fake_Bf = self.feature_extractor(self.fake_B).cuda()
        pred_fake1 = torch.cat((self.real_Bf,self.fake_Bf),1)
        self.contentLoss = self.criterionGAN(pred_fake1,True)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_Huber + self.contentLoss + + l1_reg
        self.loss_G.backward(retain_graph=retain_graph)

    def optimize_parameters(self):
        self.forward() # compute fake image G(A)
        
        # update D
        self.set_requires_grad(self.netD, True) # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradient to zero
        self.backward_D()   # calculate gradient for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G

        self.optimizer_G.step()             # udpate G's weights


