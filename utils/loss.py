import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def vae_loss(J_true, J_pred, I_true, I_pred, D_true, D_pred, mu1, logvar1, mu2, logvar2, alpha=1.0, beta=1.0):
    """
    Loss function for VAE with weighted reconstruction and KL divergence losses.

    Args:
    J_true (torch.Tensor): Ground truth input images, shape [B, 3, H, W].
    J_pred (torch.Tensor): Predicted images, shape [B, 3, H, W].
    mu1 (torch.Tensor): Mean of z1, shape [B, latent_dim].
    logvar1 (torch.Tensor): Log variance of z1, shape [B, latent_dim].
    mu2 (torch.Tensor): Mean of z2, shape [B, latent_dim].
    logvar2 (torch.Tensor): Log variance of z2, shape [B, latent_dim].
    alpha (float): Weight for the reconstruction loss term.
    beta (float): Weight for the KL divergence term.

    Returns:
    torch.Tensor: Combined loss value (scalar).
    """
    # Reconstruction loss (e.g., MSE loss)
    recon_loss = F.mse_loss(J_pred, J_true, reduction='mean')  # Sum over all elements
    I_loss = F.mse_loss(I_true, I_pred, reduction='mean')
    D_loss = F.mse_loss(D_true, D_pred, reduction='mean')

    # KL divergence for z1
    kl_loss1 = -0.5 * torch.mean(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    
    # KL divergence for z2
    kl_loss2 = -0.5 * torch.mean(1 + logvar2 - mu2.pow(2) - logvar2.exp())

    # Weighted total loss
    total_loss = alpha * (recon_loss + I_loss + D_loss) + beta * (kl_loss1 + kl_loss2)

    return total_loss


def mean_square_error(images, pred_images, weight=1):
    out = nn.MSELoss()(images * weight, pred_images * weight)

    return  torch.mean(out)

def l1_norm(images, pred_images, weight=1):
    out = nn.L1Loss()(images, pred_images)

    return  torch.mean(out) * weight

def kl_inverse_gamma(depths, pred_depths, weight=1):
    
    return weight * torch.mean(pred_depths / depths + torch.log(depths) - torch.log(pred_depths) - 1)

def kl_inverse_gaussian(depths, pred_depths, weight=1):
    
    return weight * torch.mean((depths - pred_depths) ** 2 / (depths ** 2 * pred_depths))

def cross_entropy(D_gt_bi, D_est_bi, weight=1):
    out = nn.BCEWithLogitsLoss()(D_est_bi, D_gt_bi.squeeze())

    return torch.mean(out) * weight

def distance_entropy(D_est_bi, D_gt, D_focal, lambda1):
    D_gt_bi = torch.where(D_gt >= D_focal, 1.0, 0.0).unsqueeze(1)
    entropy_D = cross_entropy(D_gt_bi, D_est_bi, lambda1)

    return entropy_D

def distance_mse(D_est_bi, D_gt, beta_est, D_focal, k_cam):
    D_est = depth_tranformer(D_est_bi, beta_est, D_focal, k_cam)
    D_gt = D_gt.unsqueeze(1)
    rmse_D = nn.MSELoss()(D_est, D_gt)

    return rmse_D

def depth_tranformer(D_est_bi, beta_est, D_focal, k_cam):
    D_est = torch.zeros_like(D_est_bi, requires_grad=False, dtype=D_est_bi.dtype, device=D_est_bi.device)
    D_est[D_est_bi >= 0] = D_focal * k_cam / (k_cam - beta_est[D_est_bi >= 0])
    D_est[D_est_bi < 0] = D_focal * k_cam / (k_cam + beta_est[D_est_bi < 0])

    return D_est


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

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


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


def tv_norm(image, weight=1):
    if len(image.shape) == 4:
        diff_i = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        diff_j = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    elif len(image.shape) == 3:
        diff_i = torch.abs(image[:, :, 1:] - image[:, :, :-1])
        diff_j = torch.abs(image[:, 1:, :] - image[:, :-1, :])

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