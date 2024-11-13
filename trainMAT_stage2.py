import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import time
# from thop import profile
# from model_part import Conv,Down,Up
# from model import Unet
from cubdl_master.example_picmus_torch import load_datasets, create_network, mk_img, dispaly_img
from options.train_options import TrainOptions
from models import create_model
from data_process import load_dataset, test_image
from util.util import diagnose_network
from models.network import UnetGenerator
import cubdl_master.PlaneWaveData
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, RandomRotation, \
    RandomVerticalFlip, RandomHorizontalFlip, Resize, ColorJitter
from metrics import image_evaluation
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from custom_data_process import CustomDataset
# from v2_custom_MATdata_process import CustomDataset
# from custom_MATdata_process import CustomDataset
# from custom_multiMATdata_process import CustomDataset #直接对数压缩
from v2_custom_multiMATdata_process import CustomDataset #对数压缩变换到正数域
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import csv
from thop import profile

# import thop

# def get_args():  # initial options for the training
#     parser = argparse.ArgumentParser(description="my code:")
#     parser.add_argument('-b', "--batch_size", type=int, default=10, help="batch size each train epoch")
#     parser.add_argument('-n', "--num_epoch", type=int, default=100, help="training epoch numbers")
#     parser.add_argument('-l', "--learning_rate", type=float, default=0.0001, help="learing rate")
#     parser.add_argument('-f', '--load', type=str, default='./img_data',
#                         help='Load model from a file')
#     parser.add_argument('-s', '--scale', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('-v', '--validation', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     return parser.parse_args()


# def makedir():  # Making the directory for saving pictures of loss change.
#     train_base = './images/' + opt.name + '/train'
#     test_base = './images/' + opt.name + '/test'
#     if not os.path.exists(train_base):
#         os.makedirs(train_base)
#     if not os.path.exists(test_base):
#         os.makedirs(test_base)
#     loss_path = './images/' + opt.name + '/train/loss.png'
#     return loss_path

def save(model, save_path, is_best=False):
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_save_path = save_path.replace('.pth', '_best.pth')
        torch.save(model.state_dict(), best_save_path)


if __name__ == '__main__':

    opt = TrainOptions().parse()  # get training options

    # Load the model
    opt.model = 'version8TriBranchPix2Pix'
    opt.name = 'tribranch_stage2'
    opt.lr = 0.0002
    model = create_model(opt)
    model.setup(opt)
    model = model.to()

    loss_file_path = '/lustre/home/c/AU/loss_file/loss_data_stage2_wm.csv'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_dirs = ['/lustre/home/c/AU/data/20240606/images_mat2']
    dataset = CustomDataset(img_dirs=img_dirs, transform=None, istrain=True)

    dataset_len = len(dataset)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    loss_G1 = 0

    lossG = np.zeros(opt.n_epochs+opt.niter_decay)
    lossD = np.zeros(opt.n_epochs+opt.niter_decay)

    with open(loss_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Iteration', 'Generator Loss', 'Discriminator Loss'])  # 写入表头

    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.niter_decay + 1):
        epoch_start_time = time.time()  # 开始计时
        train_bar = tqdm(train_loader)
        iteration_count = 0
        for batch in train_bar:

            left_images = batch['left']
            right_images = batch['right']
            mid_images = batch['mid']
            compound_images = batch['Compound']  # 这是目标图像

            n1, n2, n3 = left_images.shape
            left_images = torch.reshape(left_images, (n1, 1, n2, n3))
            right_images = torch.reshape(right_images, (n1, 1, n2, n3))
            mid_images = torch.reshape(mid_images, (n1, 1, n2, n3))
            compound_images = torch.reshape(compound_images, (n1, 1, n2, n3))


            model.set_input(left_images, right_images, mid_images, compound_images)
            model.optimize_parameters()

            model.set_input(left_images, right_images, mid_images, compound_images)
            model.training = True  # 或者 False, 取决于你想要训练模式还是测试模式
            # flops, params = profile(model, inputs=(input,))
            flops, params = profile(model.to(), inputs=())
            print(f"FLOPs: {flops}, Params: {params}")

            if total_iters % opt.print_freq == 0:
                train_bar.set_description(
                    desc='loss_G: %.4f , loss_D: %.4f' % (model.loss_G, model.loss_D))

            total_iters += 1
            # loss_G1 += model.pix2pix1.loss_G
            iteration_count += 1
            with open(loss_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, iteration_count, model.loss_G.item(), model.loss_D.item()])


        if epoch % 1 == 0:
            print(f"Saving model at epoch {epoch}")
            model.save_networks(epoch+opt.epoch)

        model.update_learning_rates()

