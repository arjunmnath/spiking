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
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("BASE_DIR"):
        base_dir = Path(os.environ.get("BASE_DIR"))
    else:
        home_dir = Path(os.path.expanduser("~"))
        base_dir = home_dir / ".cache"/ "spiking"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


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