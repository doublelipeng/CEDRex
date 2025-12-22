# github/src/__init__.py
# 暴露 model 和 dataset，使得可以直接 from github.src import LarksTransformer, Vocab ...
from .model import Transformer
from .dataset import Vocab, TransformerDataset, collate_fn
