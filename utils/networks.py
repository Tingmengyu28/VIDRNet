import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import densenet121, densenet161


def denormalize(image, mean, std):
    mean = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return image * std + mean


# Draw a sample while tarining
def visualize_sample(image, predicted_image, aif_image, predicted_aif_image, depth, predicted_depth, logger, step):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # image = denormalize(image, mean, std)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    predicted_image = (predicted_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    aif_image = (aif_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    predicted_aif_image = (predicted_aif_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # 原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Coded Image")
    axes[0, 0].axis("off")

    # 深度图标签
    # axes[1].imshow(beta.squeeze().cpu().numpy(), cmap="plasma")
    axes[0, 1].imshow(aif_image)
    axes[0, 1].set_title("Ground Truth AIF Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(depth.squeeze().cpu().numpy(), cmap="plasma")
    axes[0, 2].set_title("Ground Truth Depth")
    axes[0, 2].axis("off")

    # 预测深度图
    # axes[2].imshow(predicted_beta.squeeze().cpu().detach().numpy(), cmap="plasma")
    axes[1, 0].imshow(predicted_image)
    axes[1, 0].set_title("Predicted Coded Image")
    axes[1, 0].axis("off")

    # 深度图标签
    axes[1, 1].imshow(predicted_aif_image)
    axes[1, 1].set_title("Predicted AIF Image")
    axes[1, 1].axis("off")
    
    # 预测深度图
    axes[1, 2].imshow(predicted_depth.squeeze().cpu().detach().numpy(), cmap="plasma")
    axes[1, 2].set_title("Predicted Depth")
    axes[1, 2].axis("off")

    plt.savefig("test.png")
    logger.experiment.add_figure("Sample Visualization", fig, global_step=step)
    plt.close(fig)


class AttLayer(nn.Module):
    def __init__(self, out_chn=64, extra_chn=3):
        super(AttLayer, self).__init__()

        nf1 = out_chn // 8
        nf2 = out_chn // 4

        self.conv1 = nn.Conv2d(extra_chn, nf1, kernel_size=1, stride=1, padding=0)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf1, nf2, kernel_size=1, stride=1, padding=0)
        self.leaky2 = nn.LeakyReLU(0.2)
        self.mul_conv = nn.Conv2d(nf2, out_chn, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()
        self.add_conv = nn.Conv2d(nf2, out_chn, kernel_size=1, stride=1, padding=0)

    def forward(self, extra_maps):
        fea1= self.leaky1(self.conv1(extra_maps))
        fea2= self.leaky2(self.conv2(fea1))
        mul = self.mul_conv(fea2)
        add = self.add_conv(fea2)
        return mul, add

class AttResBlock(nn.Module):
    def __init__(self, nf=64, extra_chn=1):
        super(AttResBlock, self).__init__()
        self.extra_chn = extra_chn
        if extra_chn > 0:
            self.sft1 = AttLayer(nf, extra_chn)
            self.sft2 = AttLayer(nf, extra_chn)

        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, extra_maps):
        '''
        Input:
            feature_maps: N x c x h x w
            extra_maps: N x c x h x w or None
        '''
        mul1, add1 = self.sft1(extra_maps) if self.extra_chn > 0 else (1, 0)
        fea1 = self.conv1(self.lrelu1(feature_maps * mul1 + add1))

        mul2, add2 = self.sft2(extra_maps) if self.extra_chn > 0 else (1, 0)
        fea2 = self.conv2(self.lrelu2(fea1 * mul2 + add2))
        return torch.add(feature_maps, fea2)

class DownBlock(nn.Module):
    def __init__(self, in_chn=64, out_chn=128, extra_chn=4, n_resblocks=1, downsample=True):
        super(DownBlock, self).__init__()
        self.body = nn.ModuleList(
            [AttResBlock(in_chn, extra_chn) for _ in range(n_resblocks)]
        )
        if downsample:
            self.downsampler = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=2, padding=1)
        else:
            self.downsampler = nn.Identity()

    def forward(self, x, extra_maps):
        for op in self.body:
            x= op(x, extra_maps)
        out =self.downsampler(x)
        return out, x

class UpBlock(nn.Module):
    def __init__(self, in_chn=128, out_chn=64, n_resblocks=1):
        super(UpBlock, self).__init__()
        self.interpolater = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2, padding=0)
        self.body = nn.ModuleList([AttResBlock(nf=out_chn, extra_chn=0) for _ in range(n_resblocks)])

    def forward(self, x, bridge):
        x_up = self.interpolater(x)
        for ii, op in enumerate(self.body):
            x_up = op(x_up+bridge, None) if ii == 0 else op(x_up, None)
        return x_up


def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    return F.pad(x, pad=(0, right, 0, bottom), mode='reflect')


class AttResUNet(nn.Module):
    def __init__(self, in_chn=3,
                 extra_chn=1,
                 out_chn=1,
                 n_resblocks=2,
                 n_feat=[64, 128, 256, 512],
                 extra_mode='Input'):
        """
        Args:
            in_chn: number of input channels
            extra_chn: number of other channels, e.g., noise variance, kernel information
            out_chn: number of output channels.
            n_resblocks: number of resblocks in each scale of UNet
            n_feat: number of channels in each scale of UNet
            extra_mode: Null, Input, Down or Both
        """
        super(AttResUNet, self).__init__()

        self.out_chn = out_chn

        assert isinstance(n_feat, (tuple, list))
        self.depth = len(n_feat)

        self.extra_mode = extra_mode.lower()
        assert self.extra_mode in ['null', 'input', 'down', 'both']

        if self.extra_mode in ['down', 'null']:
            self.head = nn.Conv2d(in_chn, n_feat[0], kernel_size=3, stride=1, padding=1)
        else:
            self.head = nn.Conv2d(in_chn+extra_chn, n_feat[0], kernel_size=3, stride=1, padding=1)

        extra_chn_down = extra_chn if self.extra_mode.lower() in ['down', 'both'] else 0
        self.down_path = nn.ModuleList()
        for ii in range(self.depth):
            if ii+1 < self.depth:
                self.down_path.append(DownBlock(n_feat[ii], n_feat[ii+1],
                                                extra_chn=extra_chn_down,
                                                n_resblocks=n_resblocks,
                                                downsample=True))
            else:
                self.down_path.append(DownBlock(n_feat[ii], n_feat[ii],
                                      extra_chn=extra_chn_down,
                                      n_resblocks=n_resblocks,
                                      downsample=False))

        self.up_path = nn.ModuleList()
        for jj in reversed(range(self.depth - 1)):
            self.up_path.append(UpBlock(n_feat[jj+1], n_feat[jj], n_resblocks))

        self.tail = nn.Conv2d(n_feat[0], out_chn, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in, extra_maps_in):
        '''
        Input:
            x_in: N x [] x h x w
            extra_maps: N x []
        '''
        h, w = x_in.shape[-2:]
        x = pad_input(x_in, 2**(self.depth-1))
        if not self.extra_mode == 'null':
            extra_maps = pad_input(extra_maps_in, 2**(self.depth-1))

        if self.extra_mode in ['input', 'both']:
            x = self.head(torch.cat([x, extra_maps], 1))
        else:
            x = self.head(x)

        blocks = []
        if self.extra_mode in ['down', 'both']:
            extra_maps_down = [extra_maps,]
        for ii, down in enumerate(self.down_path):
            if self.extra_mode in ['down', 'both']:
                x, before_down = down(x, extra_maps_down[ii])
            else:
                x, before_down = down(x, None)
            if ii != len(self.down_path)-1:
                blocks.append(before_down)
                if self.extra_mode in ['down', 'both']:
                    extra_maps_down.append(F.interpolate(extra_maps, x.shape[-2:], mode='nearest'))

        for jj, up in enumerate(self.up_path):
            x = up(x, blocks[-jj-1])

        if self.out_chn != x_in.shape[1]:
            out = self.tail(x)[..., :h, :w]
        else:
            out = self.tail(x)[..., :h, :w] + x_in

        return out
