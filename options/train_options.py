from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')

        # dataset load
        # parser.add_argument('-b',"--batch_size",type=int,default=10, help="batch size each train epoch")
        # parser.add_argument('-n',"--num_epoch",type=int,default=100,help="training epoch numbers")
        # parser.add_argument('-l',"--learning_rate",type=float,default=0.0001,help="learing rate")
        parser.add_argument('-f', '--load', type=str, default='./img_data',
                        help='Load model from a file')
        parser.add_argument('-s','--scale',type=float,default=0.5,help='Downscaling factor of the images')
        parser.add_argument('-v', '--validation',type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=1, help='# of iter at starting learning rate')
        parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs with the initial learning rate')
        parser.add_argument('--niter_decay', type=int, default=1, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True # Training mode
        return parser
