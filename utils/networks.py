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
        out = torch.add(feature_maps, fea2)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_chn=64, out_chn=128, extra_chn=4, n_resblocks=1, downsample=True):
        super(DownBlock, self).__init__()
        self.body = nn.ModuleList([AttResBlock(in_chn, extra_chn) for ii in range(n_resblocks)])
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
        self.upsampler = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2, padding=0)
        self.body = nn.ModuleList([AttResBlock(nf=out_chn, extra_chn=0) for _ in range(n_resblocks)])

    def forward(self, x, bridge):
        x_up = self.upsampler(x)
        for ii, op in enumerate(self.body):
            if ii == 0:
                x_up = op(x_up+bridge, None)
            else:
                x_up = op(x_up, None)
        return x_up


def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad


class AttResUNet(nn.Module):
    def __init__(self, in_chn=3,
                 extra_chn=1,
                 out_chn=1,
                 n_resblocks=2,
                 n_feat=[64, 128, 196, 256],
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

        assert isinstance(n_feat, tuple) or isinstance(n_feat, list)
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


class ResBlock(nn.Module):
    def __init__(self, filters, filter_size, reduction=16, use_bn=False):
        """Residual Block"""
        super(ResBlock, self).__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=filter_size, stride=1, padding=filter_size // 2, bias=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=filter_size, stride=1, padding=filter_size // 2, bias=True)
        
        if use_bn:
            self.bn1 = nn.BatchNorm2d(filters)
            self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)

        return 0.2 * x + identity


class ResNet(nn.Module):
    def __init__(self, filters, filter_size, d_cnn, reduction=16, use_bn=False):
        """The CNN block to infer smooth, sparse, and noisy information"""
        super(ResNet, self).__init__()
        self.input_conv = nn.Conv2d(filters, filters, kernel_size=filter_size, stride=1, padding=filter_size // 2, bias=True)
        
        self.res_blocks = nn.ModuleList([
            ResBlock(filters, filter_size, reduction=reduction, use_bn=use_bn)
            for _ in range(d_cnn)
        ])

    def forward(self, x):
        identity = self.input_conv(x)
        out = identity
        for block in self.res_blocks:
            out = block(out)
        return out + identity
    
    
# class vae_Decoder(nn.Module):
#     """
#     The code defines a DenseVAE model that includes an encoder using DenseNet121, skip
#     connections, latent space mappings, and two decoders for image generation.
    
#     :param latent_dim: The `latent_dim` parameter represents the dimensionality of the latent
#     space in the variational autoencoder (VAE) model. It determines the size of the latent
#     vectors that encode the input data into a lower-dimensional space for learning
#     representations. In the provided code snippet, the `latent_dim` is
#     :param output_channels: The `output_channels` parameter in the `Decoder` class represents
#     the number of channels in the output image generated by the decoder. It is typically used
#     to match the number of channels in the input image that the decoder is supposed to
#     reconstruct. In the provided code snippet, `output_channels` is used
#     :param skip_connections: Skip connections in neural networks refer to connections that
#     skip one or more layers. They are used to facilitate the flow of gradients during training
#     and help in overcoming the vanishing gradient problem. In the provided code snippet,
#     skip_connections are used to pass intermediate features from the encoder to the decoder in
#     order to improve
#     :param img_size: The `img_size` parameter represents the size of the input image in terms
#     of height and width. In this case, it is a tuple `(480, 640)` indicating an image size of
#     480 pixels in height and 640 pixels in width
#     """
#     def __init__(self, latent_dim, output_channels, skip_connections, img_size):
#         super(vae_Decoder, self).__init__()
#         self.img_size = img_size
#         self.skip_connections = skip_connections  # Skip connections from the encoder
#         self.fc = nn.Linear(latent_dim, 128 * 4 * 4)  # 输出与后续解码器通道和尺寸匹配

#         # Decoder with skip connections
#         self.deconv_blocks = nn.ModuleList([
#             self._decoder_block(128, 128),  # Corresponds to 8x8 resolution
#             self._decoder_block(128, 64),  # Corresponds to 16x16 resolution
#             self._decoder_block(64, 32),  # Corresponds to 32x32 resolution
#         ])
        
#         # Final convolutional layer
#         self.final_conv = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
#         self.final_activation = nn.Sigmoid()

#     def _decoder_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, z):
#         B = z.size(0)
#         z = self.fc(z).view(B, 128, 4, 4)  # Reshape to match decoder input

#         # Apply decoder blocks with skip connections
#         for i, deconv_block in enumerate(self.deconv_blocks):
#             z = deconv_block(z)
#             if i < len(self.skip_connections):  # Apply residual connection if available
#                 skip = self.skip_connections[-(i + 1)]  # Reverse order of skip connections
#                 if skip.size() != z.size():  # Adjust size if necessary
#                     skip = F.interpolate(skip, size=z.size()[2:], mode='bilinear', align_corners=False)
#                 z = z + skip  # Residual connection

#         # Final output layer
#         z = self.final_conv(z)
#         z = F.interpolate(z, size=self.img_size, mode='bilinear', align_corners=False)
#         return self.final_activation(z)


# class DenseVAE(nn.Module):
#     def __init__(self, latent_dim, input_channels=3, img_size=(480, 640)):
#         super(DenseVAE, self).__init__()
#         self.latent_dim = latent_dim

#         # Encoder: DenseNet121
#         densenet = densenet121(pretrained=True)
#         self.encoder = nn.Sequential(*list(densenet.features.children()))
#         encoder_output_dim = 1024  # DenseNet121 最终 feature 的通道数

#         # Skip connection layers
#         self.skip_layers = [5, 7, 9]  # Select layers for skip connections (example indices)
#         self.skips = []

#         # Latent space mappings
#         self.fc_mu1 = nn.Linear(encoder_output_dim, latent_dim)
#         self.fc_logvar1 = nn.Linear(encoder_output_dim, latent_dim)
#         self.fc_mu2 = nn.Linear(encoder_output_dim, latent_dim)
#         self.fc_logvar2 = nn.Linear(encoder_output_dim, latent_dim)

#         # Decoders
#         self.decoder_I = vae_Decoder(latent_dim, input_channels, self.skips, img_size)
#         self.decoder_D = vae_Decoder(latent_dim, 1, self.skips, img_size)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         batch_size = x.size(0)
#         self.skips = []  # Reset skip connections

#         # Encoding with skip connections
#         for i, layer in enumerate(self.encoder):
#             x = layer(x)
#             if i in self.skip_layers:
#                 self.skips.append(x)  # Store intermediate features for skip connections

#         x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)  # 全局池化为 [B, 1024]

#         # Latent variables
#         mu_I, logvar_I = self.fc_mu1(x), self.fc_logvar1(x)
#         mu_D, logvar_D = self.fc_mu2(x), self.fc_logvar2(x)

#         z1 = self.reparameterize(mu_I, logvar_I)  # [B, latent_dim]
#         z2 = self.reparameterize(mu_D, logvar_D)  # [B, latent_dim]

#         # Decoding
#         I = self.decoder_I(z1)
#         D = self.decoder_D(z2)

#         return I, D, mu_I, logvar_I, mu_D, logvar_D


class vae_Decoder(nn.Module):
    """
    The code defines a VAE (Variational Autoencoder) model with a DenseNet161 encoder,
    skip connections, and two decoders for image generation.
    
    :param latent_dim: The `latent_dim` parameter in this code refers to the
    dimensionality of the latent space representation. It determines the size of the
    latent vectors that encode the input data in a lower-dimensional space. This parameter
    is used in various parts of the code, such as defining the size of the latent space,
    :param output_channels: The `output_channels` parameter refers to the number of
    channels in the output image generated by the decoder. In the provided code snippet,
    the `output_channels` parameter is used in the final convolutional layer of the
    decoder to determine the number of output channels in the generated image
    :param skip_connections: Skip connections in neural networks are connections that skip
    one or more layers. They are used to help with the flow of gradients during training
    and can improve the learning process by providing shortcuts for the gradient to flow
    through the network
    :param img_size: The `img_size` parameter represents the size of the image in pixels.
    In this case, it is a tuple `(480, 640)` indicating the image size of 480 pixels in
    height and 640 pixels in width
    """
    def __init__(self, latent_dim, output_channels, skip_connections, img_size):
        super(vae_Decoder, self).__init__()
        self.img_size = img_size
        self.skip_connections = skip_connections  # Skip connections from the encoder
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)  # 增加初始特征图的通道数

        # Decoder with more layers and channels
        self.deconv_blocks = nn.ModuleList([
            self._decoder_block(256, 256),  # 8x8
            self._decoder_block(256, 128),  # 16x16
            self._decoder_block(128, 128),  # 32x32
            self._decoder_block(128, 64),  # 64x64
            self._decoder_block(64, 32),  # 128x128
        ])
        
        # Final convolutional layer
        self.final_conv = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def _decoder_block(self, in_channels, out_channels):
        """A block with a ConvTranspose2d layer followed by BatchNorm and ReLU."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, z):
        B = z.size(0)
        z = self.fc(z).view(B, 256, 4, 4)  # Reshape to match decoder input

        # Apply decoder blocks with skip connections
        for i, deconv_block in enumerate(self.deconv_blocks):
            z = deconv_block(z)
            if i < len(self.skip_connections):  # Apply residual connection if available
                skip = self.skip_connections[-(i + 1)]  # Reverse order of skip connections
                if skip.size() != z.size():  # Adjust size if necessary
                    skip = F.interpolate(skip, size=z.size()[2:], mode='bilinear', align_corners=False)
                z = z + skip  # Residual connection

        # Final output layer
        z = self.final_conv(z)
        z = F.interpolate(z, size=self.img_size, mode='bilinear', align_corners=False)
        return self.final_activation(z)


class DenseVAE(nn.Module):
    def __init__(self, latent_dim, input_channels=3, img_size=(480, 640)):
        super(DenseVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: DenseNet161 (larger model)
        densenet = densenet161(pretrained=True)
        self.encoder = nn.Sequential(*list(densenet.features.children()))
        encoder_output_dim = 2208  # DenseNet161 最终 feature 的通道数

        # Skip connection layers
        self.skip_layers = [5, 7, 10]  # Select layers for skip connections (example indices)
        self.skips = []

        # Latent space mappings
        self.fc_mu1 = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar1 = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_mu2 = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar2 = nn.Linear(encoder_output_dim, latent_dim)

        # Decoders with increased complexity
        self.decoder_I = vae_Decoder(latent_dim, input_channels, self.skips, img_size)
        self.decoder_D = vae_Decoder(latent_dim, 1, self.skips, img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        self.skips = []  # Reset skip connections

        # Encoding with skip connections
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_layers:
                self.skips.append(x)  # Store intermediate features for skip connections

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)  # 全局池化为 [B, 2208]

        # Latent variables
        mu_I, logvar_I = self.fc_mu1(x), self.fc_logvar1(x)
        mu_D, logvar_D = self.fc_mu2(x), self.fc_logvar2(x)

        z1 = self.reparameterize(mu_I, logvar_I)  # [B, latent_dim]
        z2 = self.reparameterize(mu_D, logvar_D)  # [B, latent_dim]

        # Decoding
        I = self.decoder_I(z1)
        D = self.decoder_D(z2)

        return I, D, mu_I, logvar_I, mu_D, logvar_D