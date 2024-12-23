import torch
import torchvision.transforms as transforms
import numpy as np
import os
import h5py
import cv2
import pytorch_lightning as pl
from path import Path
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from scipy import io


class NYUDepthDataset(Dataset):
    def __init__(self, args, idx, image_size):
        
        self.args = args
        self.image_size = image_size
        
        _h5py = h5py.File('data/nyuv2/nyu_depth_v2_labeled.mat', 'r')
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
        
        aif_image = torch.from_numpy(aif_image).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).float() / int(self.args['depth_max'])
        
        if self.image_size:
            resize = transforms.Resize(size=self.image_size)
            aif_image = resize(aif_image)
            depth = resize(depth.unsqueeze(0)).squeeze(0)
        else:
            aif_image = aif_image

        aif_image.requires_grad = False
        depth.requires_grad = False

        return aif_image, depth


class NYUDepthDataModule(pl.LightningDataModule):
    def __init__(self, args, image_size, batch_size=16):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        splits = io.loadmat('data/nyuv2/splits.mat')
        train_idx = np.array(splits['trainNdxs']).squeeze(-1) - 1
        test_idx = np.array(splits['testNdxs']).squeeze(-1) - 1
        
        self.train_loader = NYUDepthDataset(self.args, train_idx, image_size=self.image_size)
        self.val_loader = NYUDepthDataset(self.args, test_idx, image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=self.args['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, shuffle=False, num_workers=self.args['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, shuffle=False, num_workers=self.args['num_workers'])
    

class Make3DDataset(Dataset):
    def __init__(self, depth_paths, image_paths, image_size, args):
        self.args = args
        self.depth_paths = depth_paths
        self.image_paths = image_paths
        self.image_size = image_size
        self.depth_files = sorted([f for f in os.listdir(self.depth_paths) if f.endswith('.mat')], key=lambda p: p.split('/')[-1].split('depth_sph_corr-')[-1])
        self.image_files = sorted([f for f in os.listdir(self.image_paths) if f.endswith('.jpg')], key=lambda p: p.split('/')[-1].split('img-')[-1])
        assert len(self.depth_files) == len(self.image_files), "The number of .mat and .jpg files should be the same."

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, index):
        depth_path = os.path.join(self.depth_paths, self.depth_files[index])
        image_path = os.path.join(self.image_paths, self.image_files[index])

        depth = loadmat(depth_path)['Position3DGrid'][:, :, 3]
        image = cv2.imread(image_path)
        depth = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        depth = torch.from_numpy(depth).float() / int(self.args['depth_max'])
        if self.transform:
            image = self.transform(image)
        
        return image, depth


class Make3DDataModule(pl.LightningDataModule):
    def __init__(self, args, image_size, batch_size=16):
        super().__init__()
        self.args = args
        self.image_size = image_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = Make3DDataset("data/make3d/Train400Depth", "data/make3d/Train400Image", self.image_size, self.args)
        self.val_dataset = Make3DDataset("data/make3d/Test134Depth", "data/make3d/Test134Image", self.image_size, self.args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.args['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.args['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.args['num_workers'])


class KITTIDataset(Dataset):
    def __init__(self, args, list_file='train.txt', transform=None, scale=1):
        super(KITTIDataset, self).__init__()
        self.args = args
        self.root = Path("/home/cxhpc/data/azt/research/CV/Defocus/data/kitti/reformed")
        scene_list_path = self.root/list_file
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.scale = scale
        self.transform = transform or transforms.ToTensor()
        self.gt_depth = []
        self.org = []

        for scene in self.scenes:
            imgs = sorted(scene.files('*.jpg'))
            depths = sorted(scene.files('*.npy'))
            self.org += imgs
            self.gt_depth += depths

        assert len(self.gt_depth) == len(self.org)

    def __getitem__(self, idx):

        img_gt_depth = np.load(self.gt_depth[idx])
        img_org = cv2.imread(self.org[idx])
        H, W, _ = img_org.shape

        if self.scale > 1:
            img_gt_depth = cv2.resize(img_gt_depth, (W // self.scale, H // self.scale))
            img_org = cv2.resize(img_org, (W // self.scale, H // self.scale))

        if self.transform is not None:
            img_org, img_gt_depth = self.transform(img_org, img_gt_depth)

        img_org = torch.from_numpy(img_org).permute(2, 0, 1).float() / 255
        img_gt_depth = torch.from_numpy(img_gt_depth).float() / self.args['depth_max']
        
        img_gt_depth.requires_grad = False
        img_org.requires_grad = False

        return img_gt_depth, img_org

    def __len__(self):
        return len(self.org)
