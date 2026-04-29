from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from stem_pytorch.util import read_yaml, Singleton, NoArgInit


@dataclass
class StemConfig:
    img_size: int
    patch_size: int
    n_frames: int
    encoder_embed_dim: int
    encoder_depth: int
    encoder_num_heads: int
    decoder_embed_dim: int
    decoder_depth: int
    decoder_num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    qk_scale: Optional[float]
    drop_rate: float
    attn_drop_rate: float
    norm_layer: str
    init_values: float
    tubelet_size: int


class Downloadable(ABC):

    @property
    @abstractmethod
    def full_model_url(self) -> str:
        pass

    @property
    @abstractmethod
    def encoder_model_url(self) -> str:
        pass


T = TypeVar("T", bound=StemConfig)

_configs = {}


def register_model(name: str):
    def wrapper(cls: Type[T]):
        _configs[name] = cls
        return cls

    return wrapper


class SharedConfig(StemConfig):
    img_size = 224
    patch_size = 16
    n_frames = 16
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    norm_layer = "LayerNorm"
    init_values = 0.
    tubelet_size = 2


@register_model("stem_vit_base_ytf")
@Singleton
class StemVitBaseConfig(NoArgInit, SharedConfig):
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    decoder_embed_dim = 512
    decoder_depth = 4
    decoder_num_heads = 6


def register_model_from_yaml(name: str, path: str) -> None:
    config = read_yaml(path)
    stem_config = StemConfig(
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
        qkv_bias=config["qkv_bias"],
        qk_scale=config["qk_scale"],
        drop_rate=config["drop_rate"],
        attn_drop_rate=config["attn_drop_rate"],
        norm_layer=config["norm_layer"],
        init_values=config["init_values"],
        tubelet_size=config["tubelet_size"]
    )
    _configs[name] = stem_config


def resolve_config(name: str) -> StemConfig:
    if name in _configs:
        return _configs[name]
    else:
        raise ValueError(f"Model {name} not found. Please register it first. The current registered models are: "
                         f"{_configs.keys()}")
