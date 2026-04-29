import argparse
import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from dataset.youtube_face import YoutubeFaceDataModule
from stem_pytorch.util import read_yaml
from model.stem_dfer import STEM_DFER

parser = argparse.ArgumentParser("STEM-DFER Stage 1 Pre-training")
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--data_dir", type=str, required=True, help="Path to YouTube Face dataset")
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--resume", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    config = read_yaml(args.config)

    # Calculate global batch size for learning rate scaling
    total_batch_size = args.batch_size * args.n_gpus
    base_lr = config["learning_rate"]["base"]
    scaled_lr = base_lr * total_batch_size / 256
    warmup_lr = config["learning_rate"]["warmup"] * total_batch_size / 256
    min_lr = config["learning_rate"]["min"] * total_batch_size / 256

    dm = YoutubeFaceDataModule(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        clip_frames=config["clip_frames"],
        temporal_sample_rate=config["temporal_sample_rate"],
        patch_size=config["patch_size"],
        tubelet_size=config["tubelet_size"],
        mask_percentage_target=0.9, # STEM-DFER uses 90% masking
        mask_strategy="tube",
        num_workers=args.num_workers
    )
    dm.setup()

    model = STEM_DFER(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        n_frames=config["clip_frames"],
        encoder_embed_dim=config["encoder"]["embed_dim"],
        encoder_depth=config["encoder"]["depth"],
        encoder_num_heads=config["encoder"]["num_heads"],
        decoder_embed_dim=config["decoder"]["embed_dim"],
        decoder_depth=config["decoder"]["depth"],
        decoder_num_heads=config["decoder"]["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        tubelet_size=config["tubelet_size"],
        learning_rate=scaled_lr,
        warmup_lr=warmup_lr,
        min_lr=min_lr,
        max_epochs=args.epochs,
        iter_per_epoch=len(dm.train_dataloader()),
        distributed=args.n_gpus > 1,
        name=config["model_name"]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"ckpt/{config['model_name']}",
        filename="stem-dfer-{epoch:02d}-{val_loss:.3f}",
        save_last=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        devices=args.n_gpus,
        accelerator="gpu" if args.n_gpus > 0 else "cpu",
        strategy="ddp" if args.n_gpus > 1 else None,
        callbacks=[checkpoint_callback],
        precision=32,
        log_every_n_steps=10
    )

    trainer.fit(model, dm, ckpt_path=args.resume)
