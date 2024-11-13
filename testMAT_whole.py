import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
from options.test_options_custom import TestOptions
from models import create_model
from data_process import load_dataset, test_image
from torch.utils.data import DataLoader,Dataset,TensorDataset
from metrics import image_evaluation
from cubdl_master.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img
from torchvision import transforms
# from custom_data_process import CustomDataset
from v2_custom_MATdata_process import CustomDataset
# from custom_MATdata_process import CustomDataset
# from skimage import color

def tensor_to_image(tensor):
    """
    将PyTorch张量转换为可用于显示的图像。
    """
    if tensor.ndim == 4:  # 假设形状为 (B, C, H, W)
        tensor = tensor.squeeze(0)  # 假设只显示第一个图像，去掉批次维度
    image = tensor.cpu().detach().numpy()  # 转换为numpy数组
    image = image.transpose((1, 2, 0))  # 转置为(H, W, C)
    image = image.clip(0, 1)  # 确保图像的值在[0, 1]范围内
    return image



def show_images(left_images, right_images, mid_images, compound_images, hr_final, threeAngleCompound_images, epoch):
    batch_size = left_images.size(0)
    show_size = 4

    fig, axs = plt.subplots(show_size, 6, figsize=(12, 3 * show_size),
                            gridspec_kw={'wspace': 0.10, 'hspace': 0.05})

    extent = [-19.0, 19.0, 60.0, 5]

    for i in range(show_size):
        images = [left_images[i], mid_images[i], right_images[i],
                  threeAngleCompound_images[i], hr_final[i], compound_images[i]]
        processed_images = []

        for img in images:
            img_np = img.squeeze().cpu().numpy()
            max_val = np.max(img_np)
            img_np = img_np - max_val
            img_np[img_np < -60] = -60
            processed_images.append(img_np)

        titles = ["Left", "Middle", "Right", "3 Plane waves", "Proposed", "75 Plane Waves"]
        for j, (img_proc, title) in enumerate(zip(processed_images, titles)):
            axs[i, j].imshow(img_proc, cmap='gray', vmin=-60, vmax=0, extent=extent, origin='upper')
            if i == 0:
                axs[i, j].set_title(title, fontsize=9)
            axs[i, j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                  labelbottom=False, labelleft=False)
            if i == show_size - 1:
                axs[i, j].tick_params(labelbottom=True, bottom=True, size=0)
            if j == 0:
                axs[i, j].tick_params(labelleft=True, left=True, size=0)

    plt.suptitle(f'Epoch {epoch} Results', fontsize=14, y=0.93)  # 设置整体标题位置和字体大小
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局来适应标题
    plt.show()


def create_circular_mask(h, w, center=None, radius=None):
    """创建一个圆形掩膜"""
    if center is None:  # 使用图像中心作为默认中心
        center = (int(w/2), int(h/2))
    if radius is None:  # 使用最小尺寸作为默认半径
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def evaluateCNR(input):
    image = input.squeeze().cpu().numpy()

    # 确保图像是灰度的
    # if image.ndim == 3:
    #     image = color.rgb2gray(image)  # 如果是RGB图像则转为灰度

    inside_center, inside_radius = (192, 127), 35
    outside_center, outside_radius = (200, 400), 40

    inside_mask = create_circular_mask(*image.shape, center=inside_center, radius=inside_radius)
    outside_mask = create_circular_mask(*image.shape, center=outside_center, radius=outside_radius)

    # 提取区域内的像素值
    inside_values = image[inside_mask]
    outside_values = image[outside_mask]

    # 计算内外部的平均值和方差
    inside_mean = np.mean(inside_values)
    outside_mean = np.mean(outside_values)
    inside_var = np.var(inside_values)
    outside_var = np.var(outside_values)

    # 计算 CNR
    cnr = (inside_mean - outside_mean) / np.sqrt((inside_var + outside_var) / 2)
    cnr_db = 20 * np.log10(abs(cnr))

    # 输出 CNR 值
    print(f'CNR (in dB): {cnr_db:.2f}')

    # 显示图像和选定区域
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    inside_circle = plt.Circle(inside_center, inside_radius, color='r', fill=False)
    outside_circle = plt.Circle(outside_center, outside_radius, color='g', fill=False)
    ax.add_patch(inside_circle)
    ax.add_patch(outside_circle)
    plt.show()

if __name__ == '__main__':


    opt = TestOptions().parse()  # get test options
    opt.name = 'tribranch_whole'
    opt.model = 'TriBranchCycleGAN'
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.epoch = 4

    if not hasattr(opt, 'use_sab'):
        opt.use_sab = False

    model = create_model(opt)
    model.setup(opt)


    if opt.eval:
        model.eval()

    model = model.to()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(img_dir='G:\\LYJ\\20240509\\images_60dB_mat_test_resolution', transform=transform, istrain = False)

    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)


    test_bar = tqdm(test_loader)


    for batch in test_bar:
        left_images = batch['left']
        right_images = batch['right']
        mid_images = batch['mid']
        compound_images = batch['Compound']
        threeAngleCompound_images = batch['ThreeAngleCompound']

        n1, n2, n3 = left_images.shape
        left_images = torch.reshape(left_images, (n1, 1, n2, n3))
        right_images = torch.reshape(right_images, (n1, 1, n2, n3))
        mid_images = torch.reshape(mid_images, (n1, 1, n2, n3))
        compound_images = torch.reshape(compound_images, (n1, 1, n2, n3))
        threeAngleCompound_images = torch.reshape(threeAngleCompound_images, (n1, 1, n2, n3))

        left_images = left_images.to(model.device)
        right_images = right_images.to(model.device)
        mid_images = mid_images.to(model.device)
        compound_images = compound_images.to(model.device)
        threeAngleCompound_images = threeAngleCompound_images.to(model.device)


        model.set_input(left_images, right_images, mid_images, compound_images)

        model.test()

        show_images(left_images, right_images, mid_images, compound_images, model.fake_Compound, threeAngleCompound_images, opt.epoch)
