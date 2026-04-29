from typing import Optional, Union, Sequence, Dict, Literal, Any

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Identity
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC

from model.stem_dfer import STEM_DFER
from stem_pytorch.config import resolve_config


class D3E_Classifier(LightningModule):
    """
    Dynamic Expression Expert Encoder (D3E) Classifier for STEM-DFER Stage 2.
    """

    def __init__(self, 
        num_classes: int, 
        backbone_config_name: str, 
        stem_ckpt: Optional[str] = None,
        deem_layers: Sequence[int] = (6, 7, 8, 9, 10, 11), # defaults to last 6 layers
        num_experts: int = 8,
        top_r: int = 2,
        reduction_dim: int = 64,
        task: Literal["multiclass", "multilabel"] = "multiclass",
        learning_rate: float = 1e-4, 
        weight_decay: float = 0.05,
        alpha_balance: float = 0.01,
        distributed: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1. Load STEM-DFER Pre-trained Encoder
        if stem_ckpt is not None:
            # Load from checkpoint
            state_dict = torch.load(stem_ckpt, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Create model instance to load weights
            # We only need the encoder part eventually
            # For now, let's create the full STEM_DFER to load easily if it matched
            config = resolve_config(backbone_config_name)
            base_model = STEM_DFER(
                img_size=config.img_size,
                patch_size=config.patch_size,
                n_frames=config.n_frames,
                encoder_embed_dim=config.encoder_embed_dim,
                encoder_depth=config.encoder_depth,
                encoder_num_heads=config.encoder_num_heads,
                tubelet_size=config.tubelet_size
            )
            base_model.load_state_dict(state_dict, strict=False)
            self.model = base_model.encoder
        else:
            # Initialize fresh if no checkpoint
            config = resolve_config(backbone_config_name)
            from stem_pytorch.model.encoder import STEMEncoder
            self.model = STEMEncoder(
                img_size=config.img_size,
                patch_size=config.patch_size,
                n_frames=config.n_frames,
                embed_dim=config.encoder_embed_dim,
                depth=config.encoder_depth,
                num_heads=config.encoder_num_heads,
                tubelet_size=config.tubelet_size
            )

        # 2. Inject DEEM modules (D3E adaptation)
        self.model.add_deem(
            layer_indices=deem_layers,
            num_experts=num_experts,
            top_r=top_r,
            reduction_dim=reduction_dim
        )
        
        # 3. Freeze base parameters and only train DEEM + Head?
        # The paper says: "仅更新少量任务相关参�?..在Transformer编码器层�?..特征同时进入原始前馈网络和DEEM模块"
        # "仅更新少量任务相关参�? usually implies freezing the pre-trained encoder.
        self._freeze_encoder_base()

        self.head = Linear(self.model.embed_dim, num_classes)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha_balance = alpha_balance
        self.distributed = distributed
        self.task = task
        
        if task == "multiclass":
            self.loss_fn = CrossEntropyLoss()
            self.acc_fn = Accuracy(task=task, num_classes=num_classes)
            self.auc_fn = AUROC(task=task, num_classes=num_classes)
        else:
            # multilabel
            from torch.nn import BCEWithLogitsLoss
            self.loss_fn = BCEWithLogitsLoss()
            self.acc_fn = Accuracy(task="binary")
            self.auc_fn = AUROC(task="binary")

    def _freeze_encoder_base(self):
        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze DEEM modules
        for block in self.model.blocks:
            if hasattr(block, "deem") and block.deem is not None:
                for param in block.deem.parameters():
                    param.requires_grad = True

    def forward(self, x):
        # Use extract_features which does PatchEmbed + PosEmbed + Blocks + Norm
        feat = self.model.extract_features(x, seq_mean_pool=True)
        return self.head(feat)

    def step(self, batch) -> Dict[str, Tensor]:
        x, y = batch
        y_hat = self(x)
        
        cls_loss = self.loss_fn(y_hat, y)
        
        # Add Load Balancing Loss
        balance_loss = self.model.get_balancing_loss()
        
        total_loss = cls_loss + self.alpha_balance * balance_loss
        
        prob = y_hat.softmax(dim=-1) if self.task == "multiclass" else y_hat.sigmoid()
        acc = self.acc_fn(prob, y)
        
        return {
            "loss": total_loss, 
            "cls_loss": cls_loss, 
            "balance_loss": balance_loss,
            "acc": acc
        }

    def training_step(self, batch, batch_idx) -> Tensor:
        loss_dict = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx) -> Tensor:
        loss_dict = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def configure_optimizers(self):
        # Only parameters with requires_grad=True (DEEM + Head)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7),
                "monitor": "val_loss"
            }
        }
