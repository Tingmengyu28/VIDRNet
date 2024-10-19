import torch
import torchvision.transforms as transforms
import numpy as np
import os
import random
import h5py
import cv2 as cv
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from scipy import io


class NYUDepthDataset(Dataset):
    def __init__(self, args, all_image_id_paths, transform):
        self.args = args
        self.transform = transform
        defocused_path = "nyu2_data/blurred_n{}".format(args['focal_distance'])
        self.all_image_paths = [
            f"{defocused_path}/{image_id_path}"
            for image_id_path in all_image_id_paths
        ]
        self.all_depth_paths = [
            f"nyu2_data/depth/{depth_id_path}"
            for depth_id_path in all_image_id_paths
        ]
        self.all_aif_image_paths = [
            f"nyu2_data/clean/{image_id_path}"
            for image_id_path in all_image_id_paths
        ]
        
    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        
        img = cv.imread(self.all_image_paths[index], cv.IMREAD_COLOR)
        img = self.transform(img)
        aif_img = cv.imread(self.all_image_paths[index], cv.IMREAD_COLOR)
        aif_img = self.transform(aif_img)
        # depth = torch.tensor(depth, dtype=torch.float32).permute(1, 0).unsqueeze(0)
        depth = cv.imread(self.all_depth_paths[index], cv.IMREAD_GRAYSCALE)
        depth = torch.from_numpy(depth) / 10 / self.args['depth_max']
        beta = get_sigma_from_depth(self.args, depth)

        return img, aif_img, beta, depth


def all_images_paths(img_path, p):
    return [file for file in os.listdir(img_path) if random.random() <= p]


def get_sigma_from_depth(args, s2):
    kcam, s1 = args["kcam"], args["focal_distance"]
    if torch.is_tensor(s2):
        s = torch.abs(s1 - s2).div(s2).to(s2.device) * kcam
    elif type(s2) == np.ndarray:
        s = np.divide(np.abs(s1 - s2), s2) * kcam
    else:
        s = np.abs(s1 - s2) / (s2) * kcam
    return s + args["lowest_beta"]


class NYUDataset(Dataset):
    def __init__(self, args, idx):
        
        self.args = args
        
        _h5py = h5py.File('nyu_depth/nyu_depth_v2_labeled.mat', 'r')
        self.images_file = np.transpose(_h5py['images'], (0, 3, 2, 1))
        self.depths_file = np.transpose(_h5py['depths'], (0, 2, 1))
        _h5py.close()
        
        self.images_file = self.images_file[idx]
        self.depths_file = self.depths_file[idx]
        
    def __len__(self):
        return len(self.images_file)

    def __getitem__(self, index):
        
        depth = self.depths_file[index]
        aif_image = self.images_file[index]
        
        aif_image = torch.from_numpy(aif_image).permute(2, 0, 1).float() / 255
        depth = torch.from_numpy(depth).float() / int(self.args['depth_max'])
        
        aif_image.requires_grad = False
        depth.requires_grad = False

        return aif_image, depth


class NYUDepthDataModule(pl.LightningDataModule):
    def __init__(self, args, batch_size=16):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        # all_image_paths = all_images_paths("nyu2_data/clean", 1)
        # val_paths = all_images_paths("nyu2_data/clean", self.args["val_ratio"])
        
        splits = io.loadmat('nyu_depth/splits.mat')
        train_idx = np.array(splits['trainNdxs']).squeeze(-1) - 1
        test_idx = np.array(splits['testNdxs']).squeeze(-1) - 1
        
        # train_paths = [img_id for img_id in all_image_paths if img_id not in val_paths]
        # self.train_loader = NYUDepthDataset(self.args, train_paths, transform=self.transform)
        # self.val_loader = NYUDepthDataset(self.args, val_paths, transform=self.transform)
        
        self.train_loader = NYUDataset(self.args, train_idx)
        self.val_loader = NYUDataset(self.args, test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=self.args['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, shuffle=False, num_workers=self.args['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, shuffle=False, num_workers=self.args['num_workers'])