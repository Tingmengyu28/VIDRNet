import argparse
import torch
import time
import os
import json
import pytorch_lightning as pl
from utils.datasets import NYUDepthDataModule, Make3DDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.process import LitDDNet


class PrintAccuracyAndLossCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        val_loss = trainer.callback_metrics['val_loss']
        val_depth_loss = trainer.callback_metrics['val_rmse_depth']
        print(f"Epoch {trainer.current_epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Depth Loss {val_depth_loss:.4f}")


if __name__ == "__main__":
    with open("configs.json", "r") as f:
        args = json.load(f)

    if args['model_name'] == 'VDRNet':
        ckpt_name = f"{args['model_name']}{args['focal_distance']}-D({args['prior_depth']}):a{args['alpha']}-m{args['mu']}-g{args['gamma']}:{time.strftime('%Y-%m-%d-%H-%M-%S')}" if args['smoothness'] is False else \
                    f"{args['model_name']}{args['focal_distance']}-D({args['prior_depth']})-smooth:a{args['alpha']}-m{args['mu']}-g{args['gamma']}:{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    else:
        ckpt_name = f"{args['model_name']}{args['focal_distance']}:{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    save_path = os.path.join(f"outputs/{args['dataset'] }/ckpts", ckpt_name)
    logger = TensorBoardLogger(f"outputs/{args['dataset']}/logs", ckpt_name)
    
    args['depth_min'] = 1e-3
    if args['dataset'] == "nyuv2":
        args['depth_max'] = 10.0
        args['f_number'] = 2.8
        args['image_size'] = (480, 640)
        data_module = NYUDepthDataModule(args=args, image_size=args['image_size'], batch_size=args['batch_size'])
        # data_module = NYUDepthDataModule_v2(args=args, batch_size=args['batch_size'])
    elif args['dataset'] == "make3d":
        args['depth_max'] = 80.0
        args['image_size'] = (320, 480)
        data_module = Make3DDataModule(args=args, image_size=args['image_size'], batch_size=args['batch_size'])

    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          monitor="val_loss",
                                          save_last=True,
                                          every_n_epochs=args['save_epoch'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args["devices"],
        max_epochs=args['max_epochs'],
        logger=logger,
        precision=args['precision'],
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor, PrintAccuracyAndLossCallback()],
    )

    parser = argparse.ArgumentParser(description='Depth estimation')
    parser.add_argument('--mode', type=str, default='train', help='train or test or resume')
    pas = parser.parse_args()

    model = LitDDNet(args=args)
    if pas.mode == 'train':
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    elif pas.mode == 'test':
        model = LitDDNet.load_from_checkpoint(args['load_path'])
        trainer.test(model, data_module)
    elif pas.mode == 'resume':
        model = LitDDNet.load_from_checkpoint(args['load_path'])
        trainer.fit(model, data_module)
        trainer.test(model, data_module)