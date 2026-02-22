import io
import os
import logging
from dataclasses import asdict

import torch
import boto3
from urllib.parse import urlparse
import fsspec
from training.configs import Snapshot
from training.utils.logging import setup_default_logging

setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)



def upload_to_s3(obj, dst):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    dst = urlparse(dst)
    boto3.client("s3").upload_fileobj(buffer, dst.netloc, dst.path.lstrip("/"))


def _load_snapshot(self):
    try:
        snapshot = fsspec.open(self.config.snapshot_path)
        with snapshot as f:
            snapshot_data = torch.load(f, map_location="cpu")
    except FileNotFoundError:
        logger.info("Snapshot not found. Training model from scratch")
        return

    snapshot = Snapshot(**snapshot_data)
    self.model.load_state_dict(snapshot.model_state)
    self.optimizer.load_state_dict(snapshot.optimizer_state)
    self.epochs_run = snapshot.finished_epoch
    logger.info(f"Resumig training from snapshot at Epoch {self.epochs_run}")

def _save_snapshot(self, epoch):
    # capture snapshot
    model = self.model
    raw_model = model.module if hasattr(model, "module") else model
    snapshot = Snapshot(
        model_state=raw_model.state_dict(),
        optimizer_state=self.optimizer.state_dict(),
        finished_epoch=epoch,
    )
    # save snapshot
    snapshot = asdict(snapshot)
    upload_to_s3(snapshot, self.config.snapshot_path)
    logger.info(f"Snapshot saved at epoch {epoch}")