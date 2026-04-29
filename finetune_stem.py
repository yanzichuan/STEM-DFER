import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model.d3e_classifier import D3E_Classifier
from dataset.celebv_hq import CelebvHqDataModule # Using CelebvHq as template for DFER
from stem_pytorch.util import read_yaml
from util.seed import Seed

parser = argparse.ArgumentParser("STEM-DFER Stage 2 Fine-tuning")
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--data_dir", type=str, required=True, help="Path to DFER dataset")
parser.add_argument("--stem_ckpt", type=str, required=True, help="Path to Stage 1 checkpoint")
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--num_classes", type=int, default=7, help="Number of emotion categories")

if __name__ == '__main__':
    args = parser.parse_args()
    Seed.set(42)
    config = read_yaml(args.config)

    # Note: Using CelebvHqDataModule as a placeholder for DFER data loading
    # You should adapt this to your specific DFER dataset (e.g., FERV39K)
    dm = CelebvHqDataModule(
        args.data_dir,
        finetune=True,
        task="action", # Placeholder task
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_frames=config["clip_frames"],
        temporal_sample_rate=2
    )
    dm.setup()

    model = D3E_Classifier(
        num_classes=args.num_classes,
        backbone_config_name=config["model_name"],
        stem_ckpt=args.stem_ckpt,
        deem_layers=(6, 7, 8, 9, 10, 11),
        num_experts=8,
        top_r=2,
        task="multiclass",
        learning_rate=1e-3, # Higher LR often used for adapter/MoE
        distributed=args.n_gpus > 1
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"ckpt/finetune_{config['model_name']}",
        filename="d3e-dfer-{epoch:02d}-{val_acc:.3f}",
        save_last=True,
        monitor="val_acc",
        mode="max"
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        devices=args.n_gpus,
        accelerator="gpu" if args.n_gpus > 0 else "cpu",
        strategy="ddp" if args.n_gpus > 1 else None,
        callbacks=[checkpoint_callback],
        precision=32
    )

    trainer.fit(model, dm)
