import os
import unittest
from typing import Optional

from .test_stem_pytorch import TestSTEM-DFERPytorch


class STEM-DFERViTBase(unittest.TestCase, TestSTEM-DFERPytorch):
    MODEL_NAME: Optional[str] = "STEM-DFER_vit_base_ytf"
    MODEL_ENCODER_PATH: Optional[str] = os.path.join("test", "model", f"STEM-DFER_vit_base_ytf.encoder.pt")
    MODEL_FULL_PATH: Optional[str] = os.path.join("test", "model", "STEM-DFER_vit_base_ytf.full.pt")
    EMBEDDING_SIZE: Optional[int] = 768
