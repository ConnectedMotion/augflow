# augflow/augmentations/base.py

from typing import Optional
from augflow.utils.unified_format import UnifiedDataset

class Augmentation:
    def __init__(self, config=None):
        self.config = config or {}
    
    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        raise NotImplementedError
