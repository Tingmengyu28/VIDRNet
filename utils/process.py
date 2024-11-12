import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
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

        if self.args['model_name'] in ['DRNet', 'VIDRNet']:
            self.dnet = define_G(3, output_nc=[1], tasks=['depth'], model_name=self.args['model_name'])
            self.rnet = AttResUNet(extra_chn=1, out_chn=3)
        elif self.args['model_name'] in ['2HDEDNet']:
            self.d3net = define_G(3, model_name=self.args['model_name'])

    def forward(self, x):
        if self.args['model_name'] in ['2HDEDNet']:
            depth, aif = self.d3net(x)
            return depth, aif
        elif self.args['model_name'] in ['DRNet', 'VIDRNet']:
            depth = self.dnet(x)
            aif = self.rnet(x, depth)
            return depth, aif


    def training_step(self, batch, batch_idx):
        aif_images, depths = batch
        aperture = torch.Tensor([self.args['f_number']] * len(batch[0])).float().cuda()
        focal_length = torch.Tensor([self.args['focal_length']] * len(batch[0])).float().cuda()
        focal_distance = torch.Tensor([self.args['focal_distance']] * len(batch[0])).float().cuda()

        images, betas = self.generator(aif_images, depths, focal_distance, aperture, focal_length)

        if self.args['model_name'] == 'VIDRNet':
            output_depths = torch.abs(self.dnet(images)[0]) + self.args['depth_min']
            output_aif_images = self.rnet(images, output_depths)
            output_depths = output_depths.squeeze()
            output_images, _ = self.generator(output_aif_images, output_depths, focal_distance, aperture, focal_length)
            
            L_depth = kl_inverse_gamma(depths, output_depths, self.args['alpha'])
            L_image = mean_square_error(aif_images, output_aif_images, self.args['mu'])
            L_rec = mean_square_error(output_images, images, self.args['gamma'])
            L_additional = tv_norm(output_depths, self.args['lambda_smooth']) + (1 - cal_ssim(output_aif_images, aif_images)) * self.args['lambda_ssim']

            loss = L_depth + L_image + L_rec + L_additional
            
        elif self.args['model_name'] == 'DRNet':
            output_depths = torch.abs(self.dnet(images)[0]) + self.args['depth_min']
            output_aif_images = self.rnet(images, output_depths)
            output_depths = output_depths.squeeze()
            output_images, _ = self.generator(output_aif_images, depths, focal_distance, aperture, focal_length)
            
            L_depth = kl_inverse_gamma(depths, output_depths, self.args['mu']) + tv_norm(output_depths, self.args['lambda_smooth'])
            L_image = l1_norm(aif_images, output_aif_images, self.args['alpha']) + (1 - cal_ssim(output_aif_images, aif_images)) * self.args['lambda_ssim']
            L_rec = l1_norm(output_images, images, self.args['gamma'])
            
            loss = L_depth + L_image + L_rec

        elif self.args['model_name'] == '2HDEDNet':
            output_depths, output_aif_images = self(images)
            output_depths = output_depths[0].squeeze()
            output_images, _ = self.generator(output_aif_images, depths, focal_distance, aperture, focal_length)
            
            L_depth = l1_norm(output_depths, depths) + tv_norm(output_depths, 0.01)
            L_image = CharbonnierLoss(aif_images, output_aif_images) + (1 - cal_ssim(output_aif_images, aif_images)) * 4
            
            loss = L_depth + 0.001 * L_image

        self.log('train_loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args['max_epochs'], eta_min=self.args['min_learning_rate']
        )
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        aif_images, depths = batch
        aperture = torch.Tensor([self.args['f_number']] * len(batch[0])).float().cuda()
        focal_length = torch.Tensor([self.args['focal_length']] * len(batch[0])).float().cuda()
        focal_distance = torch.Tensor([self.args['focal_distance']] * len(batch[0])).float().cuda()

        images, betas = self.generator(aif_images, depths, focal_distance, aperture, focal_length)

        if self.args['model_name'] == '2HDEDNet':
            output_depths, output_aif_images = self(images)
            output_depths = output_depths.squeeze()
            
        elif self.args['model_name'] == 'VIDRNet':
            output_depths = torch.abs(self.dnet(images)[0]) + self.args['depth_min']
            output_aif_images = self.rnet(images, output_depths)
            output_depths = output_depths.squeeze()

        rmse_depth = self.args['depth_max'] * torch.sqrt(mean_square_error(depths, output_depths))
        ssim_image = cal_ssim(output_aif_images, aif_images)
        AbsRel, delta1, delta2, delta3 = cal_AbsRel(depths, output_depths), cal_delta(depths, output_depths, 1), cal_delta(depths, output_depths, 2), cal_delta(depths, output_depths, 3)
        self.log('focal_distance', round(self.args['focal_distance'], 3))
        self.log('test_rmse_depth', rmse_depth.round(decimals=4))
        self.log('test_ssim_aif', ssim_image.round(decimals=4))
        self.log('test_AbsRel', AbsRel.round(decimals=4))
        self.log('test_delta_1', delta1.round(decimals=4))
        self.log('test_delta_2', delta2.round(decimals=4))
        self.log('test_delta_3', delta3.round(decimals=4))
        # visualize_sample(images[0], betas[0], output_betas[0], depths[0], output_depths[0], self.logger, self.global_step)
        return rmse_depth

    def validation_step(self, batch, batch_idx):
        aif_images, depths = batch
        aperture = torch.Tensor([self.args['f_number']] * len(batch[0])).float().cuda()
        focal_length = torch.Tensor([self.args['focal_length']] * len(batch[0])).float().cuda()
        focal_distance = torch.Tensor([self.args['focal_distance']] * len(batch[0])).float().cuda()

        images, betas = self.generator(aif_images, depths, focal_distance, aperture, focal_length)

        if self.args['model_name'] == 'VIDRNet':
            output_depths = torch.abs(self.dnet(images)[0]) + self.args['depth_min']
            output_aif_images = self.rnet(images, output_depths)
            output_depths = output_depths.squeeze()
            output_images, _ = self.generator(output_aif_images, output_depths, focal_distance, aperture, focal_length)
            
            L_depth = kl_inverse_gamma(depths, output_depths, self.args['alpha'])
            L_image = mean_square_error(aif_images, output_aif_images, self.args['mu'])
            L_rec = mean_square_error(output_images, images, self.args['gamma'])
            L_additional = tv_norm(output_depths, self.args['lambda_smooth']) + (1 - cal_ssim(output_aif_images, aif_images)) * self.args['lambda_ssim']

            loss = L_depth + L_image + L_rec + L_additional
            
        elif self.args['model_name'] == 'DRNet':
            output_depths = torch.abs(self.dnet(images)[0]) + self.args['depth_min']
            output_aif_images = self.rnet(images, output_depths)
            output_depths = output_depths.squeeze()
            output_images, _ = self.generator(output_aif_images, depths, focal_distance, aperture, focal_length)
            
            L_depth = kl_inverse_gamma(depths, output_depths, self.args['mu']) + tv_norm(output_depths, self.args['lambda_smooth'])
            L_image = l1_norm(aif_images, output_aif_images, self.args['alpha']) + (1 - cal_ssim(output_aif_images, aif_images)) * self.args['lambda_ssim']
            L_rec = l1_norm(output_images, images, self.args['gamma'])
            
            loss = L_depth + L_image + L_rec
            
        elif self.args['model_name'] == '2HDEDNet':
            output_depths, output_aif_images = self(images)
            output_depths = output_depths[0].squeeze()
            output_images, betas = self.generator(output_aif_images, depths, focal_distance, aperture, focal_length)

            L_depth = l1_norm(output_depths, depths) + tv_norm(output_depths, 0.01)
            L_image = CharbonnierLoss(aif_images, output_aif_images) + (1 - cal_ssim(output_aif_images, aif_images)) * 4

            loss = L_depth + 0.001 * L_image

        self.log('val_loss', loss)
        self.log('val_rmse_depth', self.args['depth_max'] * torch.sqrt(mean_square_error(output_depths, depths)))
        self.log('val_AbsRel', cal_AbsRel(depths, output_depths))
        self.log('val_ssim_image', cal_ssim(aif_images, output_aif_images))

        if batch_idx == 8 and self.args['model_name'] in ['VIDRNet', 'DRNet', '2HDEDNet']:
            visualize_sample(images[0], output_images[0], aif_images[0], output_aif_images[0], depths[0], output_depths[0], self.logger, self.global_step)
        return loss


# class LitDDNet(pl.LightningModule):
#     def __init__(self, args: dict):
#         super(LitDDNet, self).__init__()
#         self.save_hyperparameters()
#         self.args = args

#         self.generator = GaussPSF(args['k_size'], near=args['depth_min'], far=args['depth_max'], pixel_size=args['pixel_size'], scale=4).cuda()

#         if self.args['model_name'] in ['DRNet', 'VIDRNet']:
#             self.dnet = define_G(3, output_nc=[1], tasks=['depth'], model_name=self.args['model_name'])
#             self.rnet = AttResUNet(extra_chn=1, out_chn=3)
#         elif self.args['model_name'] in ['2HDEDNet']:
#             self.d3net = define_G(3, output_nc=[1, 3], tasks=['depth', 'reconstruction'], model_name=self.args['model_name'])

#     def forward(self, x):
#         if self.args['model_name'] in ['2HDEDNet']:
#             depth, aif = self.d3net(x)
#             return depth, aif
#         elif self.args['model_name'] in ['DRNet', 'VIDRNet']:
#             depth = self.dnet(x)
#             aif = self.rnet(x, depth)
#             return depth, aif


#     def training_step(self, batch, batch_idx):
#         images, aif_images, depths = batch

#         if self.args['model_name'] == '2HDEDNet':
#             output_depths, output_aif_images = self(images)
#             output_depths = output_depths.squeeze()
            
#             L_depth = l1_norm(output_depths, depths) + tv_norm(output_depths, 0.01)
#             L_image = CharbonnierLoss(aif_images, output_aif_images) + (1 - cal_ssim(output_aif_images, aif_images)) * 4
            
#             loss = L_depth + 0.001 * L_image

#         self.log('train_loss', loss, on_step=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=self.args['learning_rate'])
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=self.args['max_epochs'], eta_min=self.args['min_learning_rate']
#         )
#         return [optimizer], [scheduler]

#     def test_step(self, batch, batch_idx):
#         aif_images, depths = batch
#         aperture = torch.Tensor([self.args['f_number']] * len(batch[0])).float().cuda()
#         focal_length = torch.Tensor([self.args['focal_length']] * len(batch[0])).float().cuda()
#         focal_distance = torch.Tensor([self.args['focal_distance']] * len(batch[0])).float().cuda()

#         images, betas = self.generator(aif_images, depths, focal_distance, aperture, focal_length)

#         if self.args['model_name'] == '2HDEDNet':
#             output_depths, output_aif_images = self(images)
#             output_depths = output_depths.squeeze()

#         rmse_depth = self.args['depth_max'] * torch.sqrt(mean_square_error(depths, output_depths))
#         ssim_image = cal_ssim(output_aif_images, aif_images)
#         AbsRel, delta1, delta2, delta3 = cal_AbsRel(depths, output_depths), cal_delta(depths, output_depths, 1), cal_delta(depths, output_depths, 2), cal_delta(depths, output_depths, 3)
#         self.log('focal_distance', round(self.args['focal_distance'], 3))
#         self.log('test_rmse_depth', rmse_depth.round(decimals=4))
#         self.log('test_ssim_aif', ssim_image.round(decimals=4))
#         self.log('test_AbsRel', AbsRel.round(decimals=4))
#         self.log('test_delta_1', delta1.round(decimals=4))
#         self.log('test_delta_2', delta2.round(decimals=4))
#         self.log('test_delta_3', delta3.round(decimals=4))
#         # visualize_sample(images[0], betas[0], output_betas[0], depths[0], output_depths[0], self.logger, self.global_step)
#         return rmse_depth

#     def validation_step(self, batch, batch_idx):
#         images, aif_images, depths = batch
#         aperture = torch.Tensor([self.args['f_number']] * len(batch[0])).float().cuda()
#         focal_length = torch.Tensor([self.args['focal_length']] * len(batch[0])).float().cuda()
#         focal_distance = torch.Tensor([self.args['focal_distance']] * len(batch[0])).float().cuda()

#         if self.args['model_name'] == '2HDEDNet':
#             output_depths, output_aif_images = self(images)
#             output_depths = output_depths.squeeze()
#             output_images, betas = self.generator(output_aif_images, depths, focal_distance, aperture, focal_length)

#             L_depth = l1_norm(output_depths, depths) + tv_norm(output_depths, 0.01)
#             L_image = CharbonnierLoss(aif_images, output_aif_images) + (1 - cal_ssim(output_aif_images, aif_images)) * 4

#             loss = L_depth + 0.001 * L_image

#         self.log('val_loss', loss)
#         self.log('val_rmse_depth', self.args['depth_max'] * torch.sqrt(mean_square_error(output_depths, depths)))
#         self.log('val_AbsRel', cal_AbsRel(depths, output_depths))
#         self.log('val_ssim_image', cal_ssim(aif_images, output_aif_images))

#         if batch_idx == 8 and self.args['model_name'] in ['VIDRNet', 'DRNet', '2HDEDNet']:
#             visualize_sample(images[0], output_images[0], aif_images[0], output_aif_images[0], depths[0], output_depths[0], self.logger, self.global_step)
#         return loss