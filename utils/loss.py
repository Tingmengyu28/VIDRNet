import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def mean_square_error(images, pred_images, weight=1):
    out = nn.MSELoss()(images * weight, pred_images * weight)

    return  torch.mean(out)


def l1_norm(images, pred_images, weight=1):
    out = nn.L1Loss()(images, pred_images)

    return  torch.mean(out) * weight


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel, mu=1.5):
    _1D_window = gaussian(window_size, mu).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window.requires_grad = False
    return window

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel)

def cal_ssim(images, pred_images, weight=1):
    ssim = SSIM()
    out = ssim(images, pred_images)

    return torch.mean(out) * weight


def total_variation(image, p=1, weight=1):
    if len(image.shape) == 4:
        diff_i = torch.pow((torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])), p)
        diff_j = torch.pow((torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])), p)
    elif len(image.shape) == 3:
        diff_i = torch.pow(torch.abs(image[:, :, 1:] - image[:, :, :-1]), p)
        diff_j = torch.pow(torch.abs(image[:, 1:, :] - image[:, :-1, :]), p)

    tv_norm = torch.mean(diff_i) + torch.mean(diff_j)
    return weight * tv_norm


def horizontal_vertical_gradient(image, weight_h, weight_v):
    diff_i = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2
    diff_j = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2
    return weight_h * torch.mean(diff_i) + weight_v * torch.mean(diff_j)


def CharbonnierLoss(images, output_images, weight=1):
    diff = images - output_images
    loss = torch.mean(torch.sqrt(diff ** 2 + 10 ** -6))
    return loss * weight

def cal_AbsRel(depths, output_depths):
    return torch.mean(torch.abs(depths - output_depths) / depths)


def cal_delta(depths, output_depths, p):
    delta1 = output_depths / depths
    delta2 = depths / output_depths
    delta = torch.max(delta1, delta2)
    threshold = 1.25 ** p
    delta_bi = torch.zeros_like(delta, requires_grad=False, device=delta.device)
    delta_bi[delta < threshold] = 1
    return torch.mean(delta_bi)


def log_mse_torch(depths, output_depths):
    
    # 确保输入是非零的以避免对数操作报错
    predicted_depth = torch.clamp(output_depths, min=1e-6)
    ground_truth_depth = torch.clamp(depths, min=1e-6)

    # 计算对数差
    log_diff = torch.log(predicted_depth) - torch.log(ground_truth_depth)

    return torch.mean(log_diff ** 2)