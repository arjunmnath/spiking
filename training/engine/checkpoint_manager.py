import threading
import os
import json
import logging
from pathlib import Path

import torch
import boto3
from urllib.parse import urlparse
import fsspec

from concurrent.futures import ThreadPoolExecutor
import shutil
import tempfile

from training.configs import Snapshot
from training.utils.ddp import get_world_size, barrier, get_rank
from training.utils.logging import setup_default_logging
from training.utils.common import get_base_dir

setup_default_logging()
logger = logging.getLogger("CheckpointManager")

def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)


class CheckpointManager:
    def __init__(self, bucket_name: str):
        self.checkpoint_dir = get_base_dir() / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3")
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._created_checkpoints = []

    def upload_to_s3(self, file_path: Path, s3_key: str | None = None):
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")

        s3_key = s3_key or file_path.name
        with open(file_path, "rb") as f:
            self.s3.upload_fileobj(
                f,
                self.bucket_name,
                s3_key,
            )
        logger.info(f"Uploaded {file_path.name} to s3://{self.bucket_name}/{s3_key}")

    def save_checkpoint(self, step, model_data, optimizer_data, meta_data, rank=0):
        check_point_path = self.checkpoint_dir / f"ckpt_{step:06d}"
        check_point_path.mkdir(parents=True, exist_ok=True)

        if rank == 0:
            model_path = check_point_path / "model.pt"
            torch.save(model_data, model_path.as_posix())
            logger.info(f"Saved model parameters to: {model_path}")

            meta_path = check_point_path / f"meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta_data, f, indent=2)

            logger.info(f"Saved metadata to: {meta_path}")
            self._created_checkpoints.append(check_point_path)

        if optimizer_data is not None:
            optimizer_path = check_point_path / f"optim_rank{rank:d}.pt"
            torch.save(optimizer_data, optimizer_path)
            logger.info(f"Saved optimizer state to: {optimizer_path}")

    def _archive_and_upload(self, checkpoint_path: Path):
        # verifying the payloads to be uploaded
        payload_filenames = list(checkpoint_path.iterdir())
        has_model = any(p.name == "model.pt" for p in payload_filenames)
        has_meta = any(p.name == "meta.json" for p in payload_filenames)
        optim_files = [p for p in payload_filenames if p.name.startswith("optim_rank")]
        if not (has_model and has_meta and len(optim_files) == get_world_size()):
            logger.warning(f"Malformed checkpoint payload (has_model: {has_model}, has_meta: {has_meta}, optim_files: {len(optim_files)})")
            return

        with self._upload_lock:
            archive_path = checkpoint_path.with_suffix(".tar.gz")
            shutil.make_archive(
                base_name=str(archive_path).replace(".tar.gz", ""),
                format="gztar",
                root_dir=checkpoint_path.parent,
                base_dir=checkpoint_path.name,
            )
            try:
                self.upload_to_s3(archive_path, archive_path.name) # upload to s3 bucket
            except Exception as e:
                logger.exception("Failed to upload checkpoint", exc_info=e)
            archive_path.unlink(missing_ok=True) # removes the archive

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        barrier()
        if exc_type is None:
            if get_rank() == 0:
                for path in self._created_checkpoints:
                    self._archive_and_upload(path)
        barrier()
        self._executor.shutdown(wait=True)