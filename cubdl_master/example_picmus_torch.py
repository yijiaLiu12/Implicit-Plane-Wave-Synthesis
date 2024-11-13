# File:       example_PICMUS.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12
import torch
import matplotlib.pyplot as plt
import numpy as np
from das_torch import DAS_PW
from cubdl_master.PlaneWaveData import PICMUSData
from PixelGrid import make_pixel_grid

eps = 2.2204e-16

def load_datasets(acq,target,dtype):
    # Load PICMUS dataset
    database_path = "./datasets"
    assert acq == "simulation" or acq == "experiments" 
    # acq = "simulation"
    # acq = "experiments"
    assert target == "resolution_distorsion" or "contrast_speckle" or "carotid_cross" or "carotid_long"
    # target = "contrast_speckle"
    # target = "resolution_distorsion"
    assert dtype == "iq" or dtype == "rf"
    # dtype = "iq"
    P = PICMUSData(database_path, acq, target, dtype) #读取数据

    return P

def create_network(P, angle_list):
    # Define pixel grid limits (assume y == 0)
    xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
    zlims = [5e-3, 55e-3]
    wvln = P.c / P.fc
    dx = wvln / 3
    dz = dx  # Use square pixels
    grid = make_pixel_grid(xlims, zlims, dx, dz)
    fnum = 1

    # Create a DAS_PW neural network for all angles, for 1 angle
    das = DAS_PW(P, grid, angle_list)
    # idx = len(P.angles) // 2  # Choose center angle for 1-angle DAS
    # das1 = DAS_PW(P, grid, idx)

    # Store I and Q components as a tuple
    iqdata = (P.idata, P.qdata)

    return das,iqdata,xlims,zlims

def mk_img(dasN,iqdata):
    # Make 75-angle image
    idasN, qdasN = dasN.forward(iqdata)
    idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
    # detach - requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad
    # cpu - 返回一个cpu存储的目标
    idasN = idasN/np.max(idasN)
    qdasN = qdasN/np.max(qdasN)
    iqN = idasN + 1j * qdasN  # Tranpose for display purposes
    iqN = iqN + eps
    bimgN = 20 * np.log10(np.abs(iqN))  # Log-compress
    # bimgN -= np.amax(bimgN)  # Normalize by max value

    # Make 1-angle image
    # idas1, qdas1 = das1.forward(iqdata)
    # idas1, qdas1 = idas1.detach().cpu().numpy(), qdas1.detach().cpu().numpy()
    # iq1 = idas1 + 1j * qdas1  # Transpose for display purposes
    # bimg1 = 20 * np.log10(np.abs(iq1))  # Log-compress
    # bimg1 -= np.amax(bimg1)  # Normalize by max value
    # print(bimgN.max(),bimg1.max())
    # print(bimgN.min(),bimgN.min())

    return bimgN

def dispaly_img(bimg1,bimg2,bimg3,xlims,zlims,angle_list,epoch,phase,name):
    # Display images via matplotlib
    # idx = len(P.angles) // 2
    # matplotlib.use('Agg')
    extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    extent2 = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    extent3 = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]

    plt.subplot(131)
    plt.imshow(bimg1, vmin=-60, vmax=0, cmap="gray", extent=extent, origin="upper")
    if phase == 'train':
        plt.title("%d epochs LR image" % epoch,fontsize=10)
    elif phase == 'test':
        plt.title("%d test LR image" % epoch, fontsize=10)
    plt.subplot(132)
    plt.imshow(bimg2, vmin=-60, vmax=0, cmap="gray", extent=extent2, origin="upper")
    if phase == 'train':
        plt.title("%d epochs generated image" % epoch,fontsize=10)
    elif phase == 'test':
        plt.title("%d test generated image" % epoch, fontsize=10)
    plt.subplot(133)
    plt.imshow(bimg3, vmin=-60, vmax=0, cmap="gray", extent=extent3, origin="upper")
    if phase == 'train':
        plt.title("%d epochs HR image" % epoch,fontsize=10)
    elif phase == 'test':
        plt.title("%d test HR image" % epoch, fontsize=10)


    # plt.title("Angle %d: %ddeg" % (idx, P.angles[idx] * 180 / np.pi))
    train_path = './images/' + name + '/train/' + str(epoch) + '.png'
    test_path = './images/' + name + '/test/' + str(epoch) + '_test.png'
    if phase == 'train':
        plt.savefig(train_path)
    elif phase == 'test':
        plt.savefig(test_path)

    # plt.show()

    
