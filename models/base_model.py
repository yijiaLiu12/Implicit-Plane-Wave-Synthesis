import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import network
from thop import profile


# class BaseModel(ABC,torch.nn.Module):
class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    """
    # 用来在创建模型时进行初始化
    def __init__(self, opt):
        """Initialize the BaseModel class.
        """
        # super(BaseModel, self).__init__()

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        # if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        #     torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [network.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if opt.serial_train:
            load_suffix = '%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_custom_networks_serial(load_suffix)  # 用一个可以重写的方法代替直接加载

        if not self.isTrain or opt.continue_train:
            load_suffix = '%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            # self.load_networks(load_suffix)
            # self.load_Integ_networks(load_suffix)
            self.load_custom_networks(load_suffix)  # 用一个可以重写的方法代替直接加载
        self.print_networks(opt.verbose)
        #self.diagnose_network();


    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                self.save_dir = os.path.normpath(self.save_dir)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                for key in list(state_dict.keys()):  
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def load_networks_serial(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                self.save_dir = os.path.normpath(self.save_dir)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                for key in list(state_dict.keys()):  
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def load_custom_networks(self, load_suffix):
        # 默认行为是加载标准网络
        self.load_networks(load_suffix)

    def load_custom_networks_serial(self, load_suffix):
        # 默认行为是加载标准网络
        self.load_networks_serial(load_suffix)


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not"""
        if not isinstance(nets, list):
                nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    # def save_networks(self, epoch):
    #     """Save all the networks to the disk.
    #
    #     Parameters:
    #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    #     """
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             save_filename = '%s_net_%s.pth' % (epoch, name)
    #             #self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  sabe_path=./checkpoints/unet_b002
    #             save_path = os.path.join(self.save_dir, save_filename)
    #             net = getattr(self, 'net' + name)
    #
    #             if len(self.gpu_ids) > 0 and torch.cuda.is_available():
    #                 torch.save(net.module.cpu().state_dict(), save_path)
    #                 net.cuda(self.gpu_ids[0])
    #             else:
    #                 torch.save(net.cpu().state_dict(), save_path)

    def save_networks(self, epoch, networkname=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            name (str or None) -- name of the subnetwork; if None, default names will be used
            :param networkname: name of current network model
        """
        for model_name in self.model_names:
            if isinstance(model_name, str):
                if networkname is None:
                    save_filename = '%s_net_%s.pth' % (epoch, model_name)
                else:
                    save_filename = '%s_%s_net_%s.pth' % (networkname, epoch, model_name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + model_name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # def diagnose_network(self, data, name='network'):
    #     """Calculate and print the mean of average absolute(gradients)
    #
    #     Parameters:
    #         net (torch network) -- Torch network
    #         name (str) -- the name of the network
    #     """
    #     mean = 0.0
    #     count = 0
    #
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             net = getattr(self, 'net' + name)
    #             total_ops, total_params = profile(net, (data,), verbose=False)
    #             num_params = 0
    #             # for param in net.parameters():
    #             #     if param.grad is not None:
    #             #         mean += torch.mean(torch.abs(param.grad.data))
    #             #         count += 1
    #             # if count > 0:
    #             #     mean = mean / count
    #             # # print(name)
    #             # print(mean)


        # for param in self.parameters():
        #     if param.grad is not None:
        #         mean += torch.mean(torch.abs(param.grad.data))
        #         count += 1
        # if count > 0:
        #     mean = mean / count
        # print(name)
        # print(mean)
