import os.path
import shutil
from collections import deque
from pathlib import Path
from typing import Optional, Union, Sequence, Tuple, Generator
from urllib.request import urlretrieve

import cv2
import ffmpeg
import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, MSELoss
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

from .decoder import STEMDecoder
from .encoder import STEMEncoder
from ..util import read_video, padding_video, DownloadProgressBar
from ..config import resolve_config, Downloadable
from ..face_detector import FaceXZooFaceDetector


class STEM_DFER(LightningModule):
    """
    Spatio-Temporal Emotion Modeling for Dynamic Facial Expression Recognition (STEM-DFER).
    Stage 1: Dual-branch reconstruction pre-training.
    Supports feature extraction and video inference.
    """

    def __init__(self,
        img_size=224,
        patch_size=16,
        n_frames=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        norm_layer="LayerNorm",
        init_values=0.,
        tubelet_size=2,
        motion_scales=(1, 3, 7),
        lambda_loss=0.5,
        optimizer_type: str = "AdamW",
        optimizer_eps: float = 1e-8,
        optimizer_betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.05,
        learning_rate: float = 1.5e-4,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-5,
        warmup_epochs: int = 40,
        max_epochs: int = 200,
        iter_per_epoch: int = 1000,
        distributed: bool = False,
        as_feature_extractor: bool = False,
        name: str = "STEM-DFER"
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = STEMEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )

        self.motion_scales = motion_scales
        motion_channels = 3 * len(motion_scales)
        self.as_feature_extractor = as_feature_extractor
        self.clip_frames = n_frames

        if not as_feature_extractor:
            self.decoder = STEMDecoder(
                img_size=img_size,
                patch_size=patch_size,
                n_frames=n_frames,
                embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                tubelet_size=tubelet_size,
                motion_channels=motion_channels
            )
            self.enc_dec_proj = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        else:
            self.decoder = None
            self.enc_dec_proj = None

        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

        if optimizer_type == "AdamW":
            self.optimizer_type = AdamW
        elif optimizer_type == "Adam":
            self.optimizer_type = Adam
        else:
            raise ValueError("optimizer_type must be either AdamW or Adam")

        self.optimizer_eps = optimizer_eps
        self.optimizer_betas = optimizer_betas
        self.weight_decay = weight_decay

        self.learning_rate = learning_rate
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.iter_per_epoch = iter_per_epoch
        self.lambda_loss = lambda_loss

        self.lr_scheduler_factors = self._cosine_scheduler_factors()
        self.loss_fn = MSELoss()
        self.distributed = distributed
        self.name = name

    def compute_multi_scale_frame_diff(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        diffs = []
        for s in self.motion_scales:
            shifted = torch.cat([x[:, :, :1].repeat(1, 1, s, 1, 1), x[:, :, :-s]], dim=2)
            if shifted.shape[2] > t:
                shifted = shifted[:, :, :t]
            diff = x - shifted
            diffs.append(diff)
        return torch.cat(diffs, dim=1)

    def forward(self, x, mask):
        if self.as_feature_extractor:
             raise RuntimeError("For feature extraction, please use `extract_features` or `extract_video`.")
        enc_feat = self.encoder(x, mask)
        enc_feat = self.enc_dec_proj(enc_feat)
        app_pred, mot_pred = self.decoder(enc_feat, mask)
        return app_pred, mot_pred

    def training_step(self, batch, batch_idx):
        x, mask = batch
        y_app = self._patchify(x)
        motion_x = self.compute_multi_scale_frame_diff(x)
        y_mot = self._patchify(motion_x)
        
        b, _, _ = y_app.shape
        y_app_masked = y_app[~mask].view(b, -1, y_app.shape[-1])
        y_mot_masked = y_mot[~mask].view(b, -1, y_mot.shape[-1])
        
        app_pred, mot_pred = self(x, mask)
        
        loss_app = self.loss_fn(app_pred, y_app_masked)
        loss_mot = self.loss_fn(mot_pred, y_mot_masked)
        
        loss = self.lambda_loss * loss_app + (1 - self.lambda_loss) * loss_mot
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _patchify(self, x):
        p = self.patch_size
        ts = self.tubelet_size
        y = x.unfold(2, ts, ts).unfold(3, p, p).unfold(4, p, p)
        y = rearrange(y, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
        return y

    def _cosine_scheduler_factors(self):
        warmup_schedule = np.array([])
        warmup_iters = self.warmup_epochs * self.iter_per_epoch
        if self.warmup_epochs > 0:
            warmup_schedule = np.linspace(0, self.learning_rate, warmup_iters)

        iters = np.arange(self.max_epochs * self.iter_per_epoch - warmup_iters)
        schedule = np.array(
            [self.min_lr + 0.5 * (self.learning_rate - self.min_lr) * (1 + math.cos(math.pi * i / (len(iters))))
                for i in iters])

        schedule = np.concatenate((warmup_schedule, schedule))
        values_factors = schedule[::self.iter_per_epoch] / self.learning_rate
        return values_factors

    def _cosine_scheduler_fn(self, epoch):
        return self.lr_scheduler_factors[epoch]

    def configure_optimizers(self):
        optimizer = self.optimizer_type(
            self.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_eps,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay)

        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=self._cosine_scheduler_fn
        )

        return [optimizer], [lr_scheduler]

    # --- Feature Extraction Helpers ---

    def extract_features(self, x: Tensor, keep_seq: bool = True):
        if self.training:
            return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)
        else:
            with torch.no_grad():
                return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)

    @torch.no_grad()
    def extract_video(self, video_path: str, crop_face: bool = False, sample_rate: int = 2,
        stride: int = 16,
        reduction: str = "none",
        keep_seq: bool = False,
        detector_device: Optional[str] = None
    ) -> Tensor:
        self.eval()
        features = []
        for v in self._load_video(video_path, sample_rate, stride):
            if crop_face:
                if not FaceXZooFaceDetector.inited:
                    Path(".stem").mkdir(exist_ok=True)
                    FaceXZooFaceDetector.init(
                        face_sdk_path=FaceXZooFaceDetector.install(os.path.join(".stem", "FaceXZoo")),
                        device=detector_device or self.device
                    )
                v = self._crop_face(v)
            features.append(self.extract_features(v, keep_seq=keep_seq))

        features = torch.cat(features)
        if reduction == "mean":
            return features.mean(dim=0)
        elif reduction == "max":
            return features.max(dim=0)[0]
        return features

    def _crop_face(self, v: Tensor) -> Tensor:
        v_np = (rearrange(v, "b c t h w -> (b t) h w c").cpu().numpy() * 255).astype(np.uint8)
        face_frames = []
        for i in range(v_np.shape[0]):
            face_frames.append(torch.from_numpy(FaceXZooFaceDetector.crop_face(v_np[i])[0]))
        faces = torch.stack(face_frames)
        return rearrange(faces, "(b t) h w c -> b c t h w", b=1).to(self.device) / 255

    def _load_video(self, video_path: str, sample_rate: int, stride: int) -> Generator[Tensor, None, None]:
        probe = ffmpeg.probe(video_path)
        total_frames = int(probe["streams"][0]["nb_frames"])
        if total_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True) / 255
            v = padding_video(video, self.clip_frames, "same")
            yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        else:
            cap = cv2.VideoCapture(video_path)
            deq = deque(maxlen=self.clip_frames)
            current_index = -1
            while True:
                ret, frame = cap.read()
                if not ret: break
                current_index += 1
                if current_index % sample_rate == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).permute(2, 0, 1) / 255
                    deq.append(frame)
                    if len(deq) == self.clip_frames:
                        v = torch.stack(list(deq))
                        yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                        # Optional: skip by stride
            cap.release()

    @classmethod
    def from_file(cls, model_name: str, path: str) -> "STEM_DFER":
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        as_feature_extractor = True
        for key in state_dict.keys():
            if key.startswith("decoder."):
                as_feature_extractor = False
                break

        config = resolve_config(model_name)
        model = cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            n_frames=config.n_frames,
            encoder_embed_dim=config.encoder_embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_num_heads=config.encoder_num_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=config.norm_layer,
            init_values=config.init_values,
            tubelet_size=config.tubelet_size,
            as_feature_extractor=as_feature_extractor
        )
        model.load_state_dict(state_dict, strict=False)
        return model

    @classmethod
    def from_online(cls, model_name: str, full_model: bool = False) -> "STEM_DFER":
        config = resolve_config(model_name)
        if not isinstance(config, Downloadable):
            raise ValueError(f"Model {model_name} is not downloadable.")
        # ... logic for downloading
        return cls.from_file(model_name, f".stem/{model_name}.pt")
