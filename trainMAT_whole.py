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
from v2_custom_MATdata_process import CustomDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import csv



def save(model, save_path, is_best=False):
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_save_path = save_path.replace('.pth', '_best.pth')
        torch.save(model.state_dict(), best_save_path)


if __name__ == '__main__':

    opt = TrainOptions().parse()  # get training options

    # Load the model
    opt.model = 'TriBranchCycleGAN'
    opt.name = 'tribranch_whole'
    opt.serial_train = 1
    opt.epoch = 50  #The former trained epoch#
    opt.n_epochs = 100 #Eopch number to train
    opt.lr = 0.0002
    model = create_model(opt)
    model.setup(opt)
    model = model.to()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(img_dir='/lustre/home/c/AU/data/20240606/images_mat2', transform=transform)

    dataset_len = len(dataset)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    loss_G1 = 0

    lossG = np.zeros(opt.n_epochs+opt.niter_decay)
    lossD = np.zeros(opt.n_epochs+opt.niter_decay)

    loss_file_path = '/lustre/home/c/AU/loss_file/loss_data_all_wm.csv'

    with open(loss_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Iteration', 'Generator Loss Left', 'Discriminator Loss Left', 'Generator Loss Right', 'Discriminator Loss Right', 'Generator Loss Compound', 'Discriminator Loss Compound'])  # 写入表头

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

            if total_iters % opt.print_freq == 0:
                train_bar.set_description(
                    desc='loss_G1_L: %.4f , loss_D1_L: %.4f, loss_G1_R: %.4f , loss_D1_R: %.4f, loss_G2: %.4f , loss_D2: %.4f' % (model.loss_G1_L, model.loss_D1_L,model.loss_G1_R, model.loss_D1_R, model.loss_G2, model.loss_D2))

            total_iters += 1

            iteration_count += 1
            with open(loss_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, iteration_count, model.loss_G1_L.item(), model.loss_D1_L.item(),model.loss_G1_R.item(), model.loss_D1_R.item(), model.loss_G2.item(), model.loss_D2.item()])


        if epoch % 1 == 0:
            print(f"Saving model at epoch {epoch}")
            model.save_networks(epoch)

        model.update_learning_rates()


