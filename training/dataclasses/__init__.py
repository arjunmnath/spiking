from dataclasses import dataclass
from typing import Any, Dict, Optional, OrderedDict

import torch


@dataclass
class TrainingConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None
    log_every: int = None
    bucket_name: Optional[str] = None


@dataclass
class Snapshot:
    model_state: "OrderedDict[str, torch.Tensor]"
    optimizer_state: Dict[str, Any]
    finished_epoch: int


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1


@dataclass
class DataConfig:
    embed_dir: str
    data_dir: str
    precompute: bool



@dataclass
class Snapshot:
    model_state: "OrderedDict[str, torch.Tensor]"
    optimizer_state: Dict[str, Any]
    finished_epoch: int
