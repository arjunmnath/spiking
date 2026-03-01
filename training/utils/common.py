"""
Common utilities for nanochat.
"""

import os
import re
import logging
import urllib.request
from pathlib import Path
import petname

import torch
import torch.distributed as dist

from filelock import FileLock


def get_base_dir():
    # co-locate intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("BASE_DIR"):
        base_dir = Path(os.environ.get("BASE_DIR"))
    else:
        home_dir = Path(os.path.expanduser("~"))
        base_dir = home_dir / ".cache" / "spiking"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""

    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def get_run_id():
    _id = os.getenv("RUN_ID")
    if not _id:
        _id = petname.Generate(3, "-")
        os.environ["RUN_ID"] = _id
    return _id


def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
