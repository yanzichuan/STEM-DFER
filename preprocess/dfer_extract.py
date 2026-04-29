import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from stem_pytorch import STEM_DFER
from stem_pytorch.config import resolve_config

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DFER Feature Extraction")
    parser.add_argument("--backbone", type=str, default="stem_vit_base_ytf")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    # Load pre-trained model
    # model = STEM_DFER.from_online(args.backbone) # If online checkpoints exist
    # Here we assume user might have local checkpoint
    config = resolve_config(args.backbone)
    model = STEM_DFER(
        img_size=config.img_size,
        patch_size=config.patch_size,
        n_frames=config.n_frames,
        encoder_embed_dim=config.encoder_embed_dim,
        encoder_depth=config.encoder_depth,
        encoder_num_heads=config.encoder_num_heads,
        as_feature_extractor=True
    )
    feat_dir = args.backbone

    model.cuda()
    model.eval()

    raw_video_path = os.path.join(args.data_dir, "cropped")
    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))
    Path(os.path.join(args.data_dir, feat_dir)).mkdir(parents=True, exist_ok=True)
    
    for video_name in tqdm(all_videos):
        video_path = os.path.join(raw_video_path, video_name)
        save_path = os.path.join(args.data_dir, feat_dir, video_name.replace(".mp4", ".npy"))
        try:
            feat = model.extract_video(
                video_path, crop_face=False,
                sample_rate=config.tubelet_size, stride=config.n_frames,
                keep_seq=False, reduction="none")

            np.save(save_path, feat.cpu().numpy())
        except Exception as e:
            print(f"Video {video_path} error.", e)
