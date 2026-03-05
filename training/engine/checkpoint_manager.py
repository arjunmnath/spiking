import os
import json
import logging
from collections import OrderedDict
from pathlib import Path
import re

from pprint import pprint
import torch
import boto3
import tempfile
import tarfile
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor
import shutil

from training.utils.ddp import get_world_size, barrier, get_rank, is_main_process
from training.utils.logging import setup_default_logging
from training.utils.common import get_base_dir, get_run_id

setup_default_logging()
logger = logging.getLogger("CheckpointManager")


def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)


class CheckpointManager:
    def __init__(self, bucket_name: str):
        self.checkpoints_dir = get_base_dir() / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3")
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._created_checkpoint = None

    def _upload_to_s3(self, file_path: Path, s3_key: str | None = None):
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

    def save_checkpoint(self, model_data, optimizer_data, meta_data, step, rank=0):
        assert (
            type(model_data) == OrderedDict and type(optimizer_data) == OrderedDict
        ), "Expected a state dict"
        check_point_path = self.checkpoints_dir / f"{get_run_id()}_{step:06d}"
        check_point_path.mkdir(parents=True, exist_ok=True)
        self._created_checkpoint = check_point_path
        if is_main_process():
            model_path = check_point_path / "model.pt"
            torch.save(model_data, model_path.as_posix())
            logger.info(f"Saved model parameters to: {model_path}")

            meta_path = check_point_path / f"meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta_data, f, indent=2)

            logger.info(f"Saved metadata to: {meta_path}")

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
            logger.warning(
                f"Malformed checkpoint payload (has_model: {has_model}, has_meta: {has_meta}, optim_files: {len(optim_files)})"
            )
            return

        archive_path = checkpoint_path.with_suffix(".tar.gz")
        shutil.make_archive(
            base_name=str(archive_path).replace(".tar.gz", ""),
            format="gztar",
            root_dir=checkpoint_path.parent,
            base_dir=checkpoint_path.name,
        )
        try:
            self._upload_to_s3(archive_path, archive_path.name)  # upload to s3 bucket
        except Exception as e:
            logger.exception("Failed to upload checkpoint", exc_info=e)
        archive_path.unlink(missing_ok=True)  # removes the archive

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        barrier()
        if exc_type is None:
            if is_main_process():
                self._archive_and_upload(self._created_checkpoint)
        barrier()
        self._executor.shutdown(wait=True)

    def build_model(
        self,
        model,
        model_config_template,
        checkpoint_dir,
        device,
        phase,
        dirty_load=False,
        rank=0,
    ):
        assert phase in ["train", "eval"], f"Invalid phase: {phase}"
        model_data, optimizer_data, meta_data = self._load_checkpoints(
            checkpoint_dir,
            device,
            dirty_load=dirty_load,
            load_optimizer=False,
            rank=rank,
        )
        if device.type in {"cpu", "mps"}:
            model_data = {
                k: v.float() if v.dtype == torch.bfloat16 else v
                for k, v in model_data.items()
            }
        model_config_kwargs = meta_data["model_config"]
        log0(f"Building model with config: {model_config_kwargs}")
        model_config = model_config_template(**model_config_kwargs)

        with torch.device("meta"):
            model = model(model_config)
        model.to_empty(device=device)
        model.load_state_dict(model_data, strict=True, assign=True)

        if phase == "eval":
            model.eval()
        else:
            model.train()
        return model, meta_data

    def build_model_from_run_id(
        self,
        model,
        model_config_template,
        run_id,
        device,
        phase,
        dirty_load=False,
        rank=0,
    ):
        assert phase in ["train", "eval"]
        model_tag = self._find_largest_model(run_id)
        assert model_tag is not None, "no model tag found"
        ckpt_archive_path = self._download_file_with_lock(model_tag)
        ckpt_path = self.checkpoints_dir / model_tag.replace(".tar.gz", "")

        assert (
            ckpt_archive_path.exists()
            and ckpt_archive_path.is_file()
            and ckpt_archive_path.suffix == ".gz"
        )
        if not ckpt_path.exists():
            with tarfile.open(ckpt_archive_path, "r:gz") as tar:
                tar.extractall(path=self.checkpoints_dir)
                logger.info(f"Extracted checkpoint data to {self.checkpoints_dir}")
            ckpt_archive_path.unlink()

        return self.build_model(
            model,
            model_config_template,
            self.checkpoints_dir / model_tag.replace(".tar.gz", ""),
            device,
            phase,
            dirty_load=dirty_load,
            rank=rank,
        )

    def _load_checkpoints(
        self, checkpoint_dir, device, dirty_load=False, load_optimizer=False, rank=0
    ):
        model_path = checkpoint_dir / f"model.pt"
        model_data = torch.load(
            model_path, map_location=device, weights_only=not dirty_load
        )
        optimizer_data = None
        if load_optimizer:
            optimizer_path = os.path.join(checkpoint_dir, f"optim_rank{rank:d}.pt")
            optimizer_data = torch.load(
                optimizer_path, map_location=device, weights_only=not dirty_load
            )
        meta_path = os.path.join(checkpoint_dir, f"meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        return model_data, optimizer_data, meta_data

    def _find_largest_model(self, run_id: str):
        if not run_id:
            raise RuntimeError("No run id provided")
        model_tags = [f for f in self._list_s3_files() if f.startswith(run_id)]
        if not model_tags:
            raise FileNotFoundError(f"No checkpoints found for the Run Id: {run_id}")
        candidates = []
        for model_tag in model_tags:
            match = re.match(r"^([a-zA-Z-]+)_(\d+)\.tar\.gz", model_tag)
            if match:
                model_depth = int(match.group(2))
                candidates.append((model_depth, model_tag))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return None

    def _list_s3_files(self):
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)
        return (
            list(map(lambda x: x["Key"], response["Contents"]))
            if "Contents" in response
            else []
        )

    def _download_file_with_lock(self, object_key: str):
        temp_dir_path = get_base_dir() / "downloads"
        download_path = temp_dir_path / object_key
        lock_file = temp_dir_path / "download.lock"
        lock = FileLock(lock_file)
        with lock:
            self.s3.download_file(self.bucket_name, object_key, download_path)
            logger.info(
                "downloaded <b>{}</b> to <b>{}</b>".format(object_key, download_path)
            )
            return download_path
