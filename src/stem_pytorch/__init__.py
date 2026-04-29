import os.path

from .model import STEM_DFER

__all__ = [
    "STEM_DFER",
]

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file:
    __version__ = file.read()
