import argparse
import torch
import time
import os
import json
import pytorch_lightning as pl
from utils.datasets import NYUDepthDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.process import LitDDNet


if __name__ == "__main__":
    with open("configs.json", "r") as f:
        args = json.load(f)
    ckpt_name = f"{args['model_name']}{args['focal_distance']}:gamma{args['gamma']}_mu{args['mu']}_alpha{args['alpha']}:{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    save_path = os.path.join("outputs/ckpts", ckpt_name)
    data_module = NYUDepthDataModule(args=args, batch_size=args['batch_size'])
    logger = TensorBoardLogger("outputs/logs", ckpt_name)
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
        callbacks=[checkpoint_callback, lr_monitor],
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