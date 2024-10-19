import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models


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

# U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_chn=1, out_chn=1):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_chn, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_chn, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(kernel_size=2, stride=2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(kernel_size=2, stride=2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(kernel_size=2, stride=2)(enc3))

        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2, stride=2)(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


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

        if self.out_chn < x_in.shape[1]:
            out = self.tail(x)[..., :h, :w]
        else:
            out = self.tail(x)[..., :h, :w] + x_in

        return out


class CustomDenseNetEncoder(nn.Module):
    def __init__(self):
        super(CustomDenseNetEncoder, self).__init__()

        # Load the pre-trained DenseNet121 model
        densenet = models.densenet121(pretrained=True)

        # Modify the first convolution layer
        self.features = densenet.features
        self.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Create a list to store the outputs of each layer
        self.output_layers = []

        # Add the Dense Blocks and Transition Layers
        self.output_layers.append(self.features.conv0)  # Output: 64x320x240

        self.dense_block1 = self.features.denseblock1  # Block 1
        self.transition1 = self._make_transition_layer(256, 128)  # Transition 1
        self.output_layers.append(self.transition1)  # Output: 128x160x120

        self.dense_block2 = self.features.denseblock2  # Block 2
        self.transition2 = self._make_transition_layer(512, 256)  # Transition 2
        self.output_layers.append(self.transition2)  # Output: 256x80x60

        self.dense_block3 = self.features.denseblock3  # Block 3
        self.transition3 = self._make_transition_layer(1024, 512)  # Transition 3
        self.output_layers.append(self.transition3)  # Output: 512x40x30

        self.dense_block4 = self.features.denseblock4  # Block 4
        self.transition4 = self._make_transition_layer(1024, 1024)  # Transition 4
        self.output_layers.append(self.transition4)  # Output: 1024x20x15

    def _make_transition_layer(self, in_channels, out_channels):
        """Helper function to create a transition layer with Conv2d and AvgPool2d."""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        outputs = []
        # Pass through the initial layer
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        outputs.append(x)

        # Pass through each dense block and transition layer
        x = self.dense_block1(x)
        x = self.transition1(x)  # Output: 128x160x120
        outputs.append(x)

        x = self.dense_block2(x)
        x = self.transition2(x)  # Output: 256x80x60
        outputs.append(x)

        x = self.dense_block3(x)
        x = self.transition3(x)  # Output: 512x40x30
        outputs.append(x)

        x = self.dense_block4(x)
        x = self.transition4(x)  # Output: 1024x20x15
        outputs.append(x)

        return outputs  # Return all outputs for skip connections


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        :param input_channels: dimension of input channels
        :param output_channels: dimension of output channels
        """
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthEstimationDecoder(nn.Module):
    def __init__(self, input_channels):
        """
        :param input_channels: dimension of input channels
        """
        super(DepthEstimationDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            DecoderBlock(input_channels, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            # Final layer to get the depth map with 1 channel
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x, skip_connections):
        index = len(skip_connections) - 1
        for layer in self.decoder.children():
            if index >= 0:
                x = layer(x) + skip_connections[index]
                index -= 1
            else:
                x = layer(x)
        return x


# Define the Deblurring Decoder (AifD)
class DeblurringDecoder(nn.Module):
    def __init__(self, input_channels):
        """
        :param input_channels: dimension of input channels
        """
        super(DeblurringDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            DecoderBlock(input_channels, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            # Final layer to get the depth map with 1 channel
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x, skip_connections):
        # x_ini = x.clone()
        index = len(skip_connections) - 1
        for layer in self.decoder.children():
            if index >= 0:
                x = layer(x) + skip_connections[index]
                index -= 1
            else:
                x = layer(x)
        return x


# Define the complete 2HDED:NET model
class TwoHeadedDepthDeblurNet(nn.Module):
    def __init__(self):
        super(TwoHeadedDepthDeblurNet, self).__init__()
        # Encoder
        self.encoder = CustomDenseNetEncoder()

        # Depth Estimation Head (DED)
        self.depth_head = DepthEstimationDecoder(input_channels=1024)  # Adjust based on encoder output

        # Deblurring Head (AifD)
        self.deblurring_head = DeblurringDecoder(input_channels=1024)  # Adjust based on encoder output

    def forward(self, x):
        # Encode the input
        encoded_features = self.encoder(x)
        x = encoded_features[-1]

        # Get depth map from DED
        depth_map = self.depth_head(x, encoded_features)

        # Get deblurred image from AifD
        deblurred_image = self.deblurring_head(x, encoded_features)

        return depth_map, deblurred_image