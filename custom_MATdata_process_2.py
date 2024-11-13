import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
import random
from scipy.io import loadmat
import h5py
import numpy as np

class RandomHorizontalFlipTransform:
    """同时对一组图像进行随机水平翻转的transform"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if random.random() < self.p:
            # # 如果随机数小于概率p，则翻转所有图像
            # return [F.hflip(image) for image in images]
            flipped_images = [F.hflip(image) for image in images]
            # 交换第一个和第二个图像，即 left_img 和 right_img
            flipped_images[0], flipped_images[1] = flipped_images[1], flipped_images[0]
            return flipped_images
        return images

# class MatRandomHorizontalFlipTransform:
#     """对一组bmode数组进行随机水平翻转，并在翻转后交换左右图像的数据，以模拟视角变换。"""
#     def __init__(self, p=0.5):
#         self.p = p  # 翻转的概率
#
#     def __call__(self, bmode_dict):
#         if random.random() < self.p:
#             # 应用水平翻转
#             flipped_bmode_dict = {key: np.fliplr(bmode) if bmode.ndim > 1 else bmode for key, bmode in bmode_dict.items()}
#             # 交换翻转后的左右图像数据
#             flipped_bmode_dict['left'], flipped_bmode_dict['right'] = flipped_bmode_dict['right'], flipped_bmode_dict['left']
#             return flipped_bmode_dict
#         return bmode_dict

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, istrain = True):
        """
        Args:
            img_dir (string): 图像文件夹的路径。
            transform (callable, optional): 一个可选的转换函数，你可以传递任何你希望的变换。
        """
        self.istrain = istrain
        self.img_dir = img_dir
        self.transform = transform
        self.flip_transform = RandomHorizontalFlipTransform(p=0.5)

        # # 加载所有图像的路径
        # self.left_images = sorted(os.listdir(os.path.join(img_dir, 'left')))
        # self.right_images = sorted(os.listdir(os.path.join(img_dir, 'right')))
        # self.mid_images = sorted(os.listdir(os.path.join(img_dir, 'mid')))
        self.compound_images = sorted(os.listdir(os.path.join(img_dir, 'Compound')))
        # self.threeAngleCompound = sorted(os.listdir(os.path.join(img_dir, 'ThreeAngleCompound')))
        # 加载所有图像的路径
        self.img_paths = {angle: sorted(os.listdir(os.path.join(img_dir, angle)))
                          for angle in ['left', 'right', 'mid', 'Compound', 'ThreeAngleCompound']}
        if not self.istrain:
            # 加载掩膜
            self.cyst_masks = sorted(os.listdir(os.path.join(img_dir, 'x_mask','cyst_mask')))
            self.background_masks = sorted(os.listdir(os.path.join(img_dir, 'x_mask','background_mask')))

    def __len__(self):
        return len(self.compound_images)

    def __getitem__(self, idx):
        data = {}
        for angle in ['left', 'right', 'mid', 'Compound', 'ThreeAngleCompound']:
            img_path = os.path.join(self.img_dir, angle, self.img_paths[angle][idx])
            with h5py.File(img_path, 'r') as file:
                img = file['log_data'][:]  # 注意，这里假设数据存储在文件的'log_data'字段
            img = torch.tensor(img, dtype=torch.float32)
            # img = torch.clamp(img, max=-0.001)  # 将所有大于0的值替换为0（matlab因为插值会导致值大于0）
            # mask = img > 0  # 创建一个掩码，标记所有正值
            # random_values = -torch.rand(mask.sum()) * 0.01  # 对于每个非负值，生成一个在0到-0.01之间的随机值
            # img[mask] = random_values  # 应用随机值
            min_value = img.min().item()
            img = img - min_value
            data[angle] = img.T #转置

        if self.istrain:
            # 只在训练时应用翻转
            stack = torch.stack([data[angle] for angle in data], dim=0)  # 将所有图像堆叠为一个新的维度
            stack = self.flip_transform(stack)  # 应用翻转变换
            data = {angle: stack[i] for i, angle in enumerate(data)}
        else:
            mask_cyst_path = os.path.join(self.img_dir, 'x_mask','cyst_mask', self.cyst_masks[idx])
            mask_background_path = os.path.join(self.img_dir, 'x_mask', 'background_mask', self.background_masks[idx])

            # data['mask_cyst'] =  Image.open(mask_cyst_path).convert('L')
            # data['mask_background'] = Image.open(mask_background_path).convert('L')
            mask_cyst = Image.open(mask_cyst_path).convert('L')
            mask_background = Image.open(mask_background_path).convert('L')

            # 将PIL图像转换为Numpy数组
            mask_cyst_array = np.array(mask_cyst)
            mask_background_array = np.array(mask_background)

            # 转换数组中的值，使其只包含0和1
            data['mask_cyst'] = (mask_cyst_array > 0).astype(int)  # 假设非0值代表掩膜区域
            data['mask_background'] = (mask_background_array > 0).astype(int)  # 假设非0值代表掩膜区域

        return data

    # def __getitem__(self, idx):
    #     left_img_path = os.path.join(self.img_dir, 'left', self.left_images[idx])
    #     right_img_path = os.path.join(self.img_dir, 'right', self.right_images[idx])
    #     mid_img_path = os.path.join(self.img_dir, 'mid', self.mid_images[idx])
    #     compound_img_path = os.path.join(self.img_dir, 'Compound', self.compound_images[idx])
    #     threeAngleCompound_img_path = os.path.join(self.img_dir, 'ThreeAngleCompound', self.threeAngleCompound[idx])
    #
    #     # left_img = Image.open(left_img_path)
    #     # right_img = Image.open(right_img_path)
    #     # mid_img = Image.open(mid_img_path)
    #     # compound_img = Image.open(compound_img_path)
    #     # threeAngleCompound_img = Image.open(threeAngleCompound_img_path)
    #
    #     mat = loadmat(left_img_path)
    #     left_img = mat['log_data']
    #     mat = loadmat(right_img_path)
    #     right_img = mat['log_data']
    #     mat = loadmat(mid_img_path)
    #     mid_img = mat['log_data']
    #     mat = loadmat(compound_img_path)
    #     compound_img = mat['log_data']
    #     mat = loadmat(threeAngleCompound_img_path)
    #     threeAngleCompound_img = mat['log_data']
    #
    #     if self.transform:
    #         left_img = self.transform(left_img)
    #         right_img = self.transform(right_img)
    #         mid_img = self.transform(mid_img)
    #         compound_img = self.transform(compound_img)
    #         threeAngleCompound_img = self.transform(threeAngleCompound_img)
    #
    #     if self.istrain:
    #         images = [left_img, right_img, mid_img, compound_img, threeAngleCompound_img]
    #         images = self.flip_transform(images)   # 统一翻转
    #
    #         left_img, right_img, mid_img, compound_img, threeAngleCompound_img = images
    #
    #
    #     return {'left': left_img, 'right': right_img, 'mid': mid_img, 'compound': compound_img, 'threeAngleCompound': threeAngleCompound_img}


# class CustomMatDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         """
#         Args:
#             data_dir (string): 数据文件夹的路径。
#             transform (callable, optional): 一个可选的转换函数，你可以传递任何你希望对bmode数组进行的变换。
#         """
#         self.data_dir = data_dir
#         self.transform = transform
#         self.flip_transform = MatRandomHorizontalFlipTransform(p=0.5)
#
#         # 加载所有.mat文件的路径
#         self.left_files = sorted(os.listdir(os.path.join(data_dir, 'left')))
#         self.right_files = sorted(os.listdir(os.path.join(data_dir, 'right')))
#         self.mid_files = sorted(os.listdir(os.path.join(data_dir, 'mid')))
#         self.compound_files = sorted(os.listdir(os.path.join(data_dir, 'Compound')))
#
#     def __len__(self):
#         return len(self.compound_files)
#
#     def __getitem__(self, idx):
#         # 从.mat文件中读取bmode数组
#         def load_bmode(file_path):
#             data = loadmat(file_path)
#             bmode = data['bmode']
#             return bmode
#
#         left_path = os.path.join(self.data_dir, 'left', self.left_files[idx])
#         right_path = os.path.join(self.data_dir, 'right', self.right_files[idx])
#         mid_path = os.path.join(self.data_dir, 'mid', self.mid_files[idx])
#         compound_path = os.path.join(self.data_dir, 'Compound', self.compound_files[idx])
#
#         left_bmode = load_bmode(left_path)
#         right_bmode = load_bmode(right_path)
#         mid_bmode = load_bmode(mid_path)
#         compound_bmode = load_bmode(compound_path)
#
#         if self.transform:
#             left_bmode = self.transform(left_bmode)
#             right_bmode = self.transform(right_bmode)
#             mid_bmode = self.transform(mid_bmode)
#             compound_bmode = self.transform(compound_bmode)
#
#         bmode_dict = {
#             'left': left_bmode,
#             'right': right_bmode,
#             'mid': mid_bmode,
#             'compound': compound_bmode
#         }
#
#         # 现在调用 flip_transform 来应用随机水平翻转
#         bmode_dict = self.flip_transform(bmode_dict)
#
#         # return {'left': left_bmode, 'right': right_bmode, 'mid': mid_bmode, 'compound': compound_bmode}
#         return {
#             'left': bmode_dict['left'],
#             'right': bmode_dict['right'],
#             'mid': bmode_dict['mid'],
#             'compound': bmode_dict['compound']
#         }
