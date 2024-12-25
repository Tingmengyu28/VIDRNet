import torch.optim as optim
import pytorch_lightning as pl
import torch.distributions as tdist
from torcheval.metrics import PeakSignalNoiseRatio
from torchvision.utils import save_image
from draw_depth import save_depth_as_image
from d3networks.networks import define_G
from utils.GaussPSFLayer import GaussPSF
from utils.loss import *
from utils.networks import *


class LitDDNet(pl.LightningModule):
    def __init__(self, args: dict):
        super(LitDDNet, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.generator = GaussPSF(args['k_size'], near=args['depth_min'], far=args['depth_max'], pixel_size=args['pixel_size'], scale=4).cuda()
        self.outputs = []
        if self.args['dataset'] == 'make3d':
            self.outputs_c1 = []

        if self.args['model_name'] == 'VDRNet':
            self.dnet = define_G(3, output_nc=[1], tasks=['depth'], pretrained=True, model_name=self.args['model_name'])
            # Recommand 'Input' for NYU dataset, 'Both' for Make3D dataset
            # if self.args['dataset'] == 'nyuv2':
            #     extra_mode = 'Input'
            # elif self.args['dataset'] == 'make3d':
            #     extra_mode = 'Both'
            self.rnet = AttResUNet(extra_chn=1, out_chn=3, extra_mode='Both')
        elif self.args['model_name'] == '2HDEDNet':
            self.d3net = define_G(3, pretrained=True, model_name=self.args['model_name'])
        elif self.args['model_name'] == 'D3Net':
            self.d3net = define_G(3, output_nc=[1], tasks=['depth'], pretrained=True, model_name=self.args['model_name'])

    def forward(self, image):
        if self.args['model_name'] == '2HDEDNet':
            return self.d3net(image)
        
        elif self.args['model_name'] == 'VDRNet':
            return self._forward_noise(image)

        elif self.args['model_name'] == 'D3Net':
            return self.d3net(image)

    # TODO Rename this here and in `forward`
    def _forward_noise(self, image):
        depth = torch.abs(self.dnet(image)[0]) + self.args['depth_min']
        aif = self.rnet(image, depth)
        depth = depth.squeeze()

        return depth, aif

    def _defocus_images_noise(self, batch):
        aif_images, depths = batch
        aperture = torch.Tensor([self.args['f_number']] * len(batch[0])).float().cuda()
        focal_length = torch.Tensor([self.args['focal_length']] * len(batch[0])).float().cuda()
        focal_distance = torch.Tensor([self.args['focal_distance']] * len(batch[0])).float().cuda()
        images, betas = self.generator(aif_images, depths, focal_distance, aperture, focal_length)
        images += torch.randn_like(images) / (self.args['alpha'] * 255.0)
        
        return images, depths, aif_images, focal_distance, aperture, focal_length

    def _kl_divergence_depth(self):
        if self.args['prior_depth'] == 'gaussian':
            kl_depth = mean_square_error
            self.p = 2
        elif self.args['prior_depth'] == 'laplacian':
            kl_depth = l1_norm
            self.p = 1
            
        return kl_depth
    
    def reparameterlize_depth(self, depths):
        if self.args['prior_depth'] == 'gaussian':
            depths_sample = tdist.Normal(depths, 1 / (self.args['alpha'] * self.args['depth_max'])).rsample()
        elif self.args['prior_depth'] == 'laplacian':
            depths_sample = tdist.laplace.Laplace(depths, 1 / (self.args['alpha'] * self.args['depth_max'])).rsample()
        elif self.args['prior_depth'] == 'gamma':
            depths_sample = tdist.gamma.Gamma(self.args['alpha'] + 1, self.args['alpha'] / depths).rsample()
        
        return depths_sample

    def _VDRNet_loss(self, depths, aif_images, images, focal_distance, aperture, focal_length):
        output_depths, output_aif_images = self(images)
        # reparameterization trick
        output_aif_images_sample = output_aif_images + torch.randn_like(output_aif_images) / (self.args['mu'] * 255.0)
        output_depths_sample = self.reparameterlize_depth(output_depths)
        output_images, _ = self.generator(output_aif_images_sample, output_depths_sample, focal_distance, aperture, focal_length)
        L_depth = self._kl_divergence_depth()(depths, output_depths, self.args['alpha'])
        L_image = mean_square_error(aif_images, output_aif_images, self.args['mu'])
        L_rec = mean_square_error(output_images, images, self.args['gamma'])
        L_additional = total_variation(output_depths, self.p, self.args['lambda_d']) + \
                       total_variation(output_aif_images, self.p, self.args['lambda_z']) if self.args['smoothness'] else 0

        return {'loss': L_depth + L_image + L_rec + L_additional,
                'output_depths': output_depths,
                'output_aif_images': output_aif_images,
                'output_images': output_images}


    def _2hdednet_loss(self, depths, aif_images, images, focal_distance, aperture, focal_length):
        output_depths, output_aif_images = self(images)
        output_depths = output_depths.squeeze()
        output_images, _ = self.generator(output_aif_images, depths, focal_distance, aperture, focal_length)
        
        L_depth = l1_norm(output_depths, depths) + total_variation(output_depths, 1, 0.01)
        L_image = CharbonnierLoss(aif_images, output_aif_images) + (1 - cal_ssim(output_aif_images, aif_images)) * 4
        
        return {'loss': L_depth + 0.001 * L_image,
                'output_depths': output_depths,
                'output_aif_images': output_aif_images,
                'output_images': output_images}
        
    def _d3net_loss(self, depths, images):
        output_depths = self(images)[0]
        output_depths = output_depths.squeeze()
        
        return {'loss': l1_norm(output_depths, depths), 
                'output_depths': output_depths}

    def training_step(self, batch, batch_idx):
        images, depths, aif_images, focal_distance, aperture, focal_length = self._defocus_images_noise(batch)
        if self.args['model_name'] == 'VDRNet':
            outputs = self._VDRNet_loss(depths, aif_images, images, focal_distance, aperture, focal_length)
            loss, _, _, _ = outputs.values()
        elif self.args['model_name'] == '2HDEDNet':
            outputs = self._2hdednet_loss(depths, aif_images, images, focal_distance, aperture, focal_length)
            loss, _, _, _ = outputs.values()
        elif self.args['model_name'] == 'D3Net':
            loss, _ = self._d3net_loss(depths, images).values()
            
        self.log('train_loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args['max_epochs'], eta_min=self.args['min_learning_rate']
        )
        return [optimizer], [scheduler]

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, depths, aif_images, _, _, _ = self._defocus_images_noise(batch)
        if self.args['model_name'] == '2HDEDNet':
            output_depths, output_aif_images = self(images)
            output_depths = output_depths.squeeze()
        elif self.args['model_name'] == 'VDRNet':
            output_depths, output_aif_images = self(images)
        elif self.args['model_name'] == 'D3Net':
            output_depths, output_aif_images = self(images)[0].squeeze(), aif_images
        
        depths, output_depths = depths * self.args['depth_max'], output_depths * self.args['depth_max']
        mse_depth = mean_square_error(depths.squeeze(), output_depths)
        log_mse = log_mse_torch(depths.squeeze(), output_depths)
        AbsRel = cal_AbsRel(depths.squeeze(), output_depths)
        delta1 = cal_delta(depths.squeeze(), output_depths, 1)
        delta2 = cal_delta(depths.squeeze(), output_depths, 2)
        delta3 = cal_delta(depths.squeeze(), output_depths, 3)
        
        ssim_image = cal_ssim(aif_images, output_aif_images)
        psnr = PeakSignalNoiseRatio(data_range=1.0)
        psnr.update(output_aif_images, aif_images)

        if batch_idx == 71:
            save_depth_as_image(depths[0], f"output_images/{self.args['dataset']}/D.png")
            save_image(aif_images[0], f"output_images/{self.args['dataset']}/I.png")
            save_image(images[0], f"output_images/{self.args['dataset']}/J.png")
            save_image(output_aif_images[0], f"output_images/{self.args['dataset']}/I_{self.args['model_name']}.png")
            if self.args['model_name'] == 'VDRNet':
                if len(output_depths.shape) == 2:
                    output_depths.unsqueeze_(0)
                save_depth_as_image(output_depths[0], f"output_images/{self.args['dataset']}/D_{self.args['model_name']}_{self.args['prior_depth']}.png")
            else:
                save_depth_as_image(output_depths[0], f"output_images/{self.args['dataset']}/D_{self.args['model_name']}.png")

        self.outputs.append({'mse_depth': mse_depth, 
                             'log_mse_depth': log_mse, 
                             'AbsRel': AbsRel, 
                             'delta_1': delta1, 
                             'delta_2': delta2, 
                             'delta_3': delta3, 
                             'psnr_aif': psnr.compute(),
                             'ssim_aif': ssim_image})
        
        if self.args['dataset'] == 'make3d' and depths.max() <= 70.0:
            self.outputs_c1.append({'mse_depth': mse_depth, 
                                    'AbsRel': AbsRel, 
                                    'psnr_aif': psnr.compute(),
                                    'ssim_aif': ssim_image})

    @torch.no_grad()
    def on_test_epoch_end(self):
        outputs = self.outputs
        rmse_depth = torch.sqrt(torch.stack([x['mse_depth'] for x in outputs]).mean())
        log_rmse_depth = torch.sqrt(torch.stack([x['log_mse_depth'] for x in outputs]).mean())
        AbsRel = torch.stack([x['AbsRel'] for x in outputs]).mean()
        delta1 = torch.stack([x['delta_1'] for x in outputs]).mean()
        delta2 = torch.stack([x['delta_2'] for x in outputs]).mean()
        delta3 = torch.stack([x['delta_3'] for x in outputs]).mean()
        psnr_aif = torch.stack([x['psnr_aif'] for x in outputs]).mean()
        ssim_aif = torch.stack([x['ssim_aif'] for x in outputs]).mean()
        
        self.log('rmse_depth', rmse_depth.round(decimals=4))
        self.log('log_rmse_depth', log_rmse_depth.round(decimals=4))
        self.log('AbsRel', AbsRel.round(decimals=4))
        self.log('delta_1', delta1.round(decimals=4))
        self.log('delta_2', delta2.round(decimals=4))
        self.log('delta_3', delta3.round(decimals=4))
        self.log('psnr_aif', psnr_aif.round(decimals=4))
        self.log('ssim_aif', ssim_aif.round(decimals=4))
        
        if self.args['dataset'] == 'make3d':
            outputs_c1 = self.outputs_c1
            if len(outputs_c1) > 0:
                rmse_depth_c1 = torch.sqrt(torch.stack([x['mse_depth'] for x in outputs_c1]).mean())
                AbsRel_c1 = torch.stack([x['AbsRel'] for x in outputs_c1]).mean()
                psnr_aif_c1 = torch.stack([x['psnr_aif'] for x in outputs_c1]).mean()
                ssim_aif_c1 = torch.stack([x['ssim_aif'] for x in outputs_c1]).mean()
                self.log('c1_rmse_depth', rmse_depth_c1.round(decimals=4))
                self.log('c1_AbsRel', AbsRel_c1.round(decimals=4))
                self.log('c1_psnr_aif', psnr_aif_c1.round(decimals=4))
                self.log('c1_ssim_aif', ssim_aif_c1.round(decimals=4))

    def validation_step(self, batch, batch_idx):
        images, depths, aif_images, focal_distance, aperture, focal_length = self._defocus_images_noise(batch)
        if self.args['model_name'] == 'VDRNet':
            outputs = self._VDRNet_loss(depths, aif_images, images, focal_distance, aperture, focal_length)
            loss, output_depths, output_aif_images, output_images = outputs.values()
        elif self.args['model_name'] == '2HDEDNet':
            outputs = self._2hdednet_loss(depths, aif_images, images, focal_distance, aperture, focal_length)
            loss, output_depths, output_aif_images, output_images = outputs.values()
        elif self.args['model_name'] == 'D3Net':
            loss, output_depths = self._d3net_loss(depths, images).values()
            output_images, _ = self.generator(aif_images, output_depths, focal_distance, aperture, focal_length)
            output_aif_images = aif_images

        self.log('val_loss', loss, sync_dist=True)
        self.log('val_rmse_depth', self.args['depth_max'] * torch.sqrt(mean_square_error(output_depths, depths)), sync_dist=True)
        self.log('val_AbsRel', cal_AbsRel(depths, output_depths), sync_dist=True)
        self.log('val_ssim_image', cal_ssim(aif_images, output_aif_images), sync_dist=True)

        if batch_idx == 2:
            visualize_sample(images[0], output_images[0], aif_images[0], output_aif_images[0], depths[0], output_depths[0], self.logger, self.global_step)
        return loss
    
    def on_validation_end(self):
        return super().on_validation_end()
