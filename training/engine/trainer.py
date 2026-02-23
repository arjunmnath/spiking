import logging
import os
from dataclasses import asdict
from typing import List, Tuple

import fsspec
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config_classes import Snapshot, TrainingConfig
from utils import upload_to_s3

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        optimizer,
        metrices,
        train_dataset: Dataset,
        test_dataset: Dataset,
        use_ddp: bool = True,
    ):
        self.config = config
        self.model = model
        self.metrices = metrices
        self.local_rank = dist.get_rank() if use_ddp else 0
        self.use_ddp = use_ddp
        self.acc = torch.accelerator.current_accelerator()
        self.device: torch.device = torch.device(f"{self.acc}:{self.local_rank}")
        self.device_type = self.device.type

        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = (
            self._prepare_dataloader(test_dataset) if test_dataset else None
        )

        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.model.apply(self.init_weights)
        self.optimizer = optimizer
        self.save_every = self.config.save_every

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=100)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs * len(self.train_loader) - 100,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[100],
        )
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()
        if self.device_type == "cuda" and self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.config.use_amp = True
        if self.config.use_amp:
            self.scaler = GradScaler(self.device_type)

    def _prepare_dataloader(self, dataset: Dataset):
        return (
            DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=self.config.data_loader_workers,
                sampler=DistributedSampler(dataset) if self.use_ddp else None,
            )
            if (self.device_type == "cuda")
            else DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=self.config.data_loader_workers,
            )
        )

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

    def _test_batch(
        self,
        history: torch.Tensor,
        clicks: torch.Tensor,
        non_clicks: torch.Tensor,
        log: bool,
    ) -> Tuple[List[float], float]:
        with torch.set_grad_enabled(False), torch.amp.autocast(
            device_type=self.device_type,
            dtype=torch.float16,
            enabled=(self.config.use_amp),
        ):
            (
                loss,
                preds,
                target,
                indexes,
                attn_scores,
                seq_len,
            ) = self.model(history, clicks, non_clicks, log=True)
            metrices = [
                metric(preds, target, indexes=indexes) for metric in self.metrices
            ]
        return metrices, loss.item()

    def _train_batch(
        self,
        history: torch.Tensor,
        clicks: torch.Tensor,
        non_clicks: torch.Tensor,
        log: bool,
    ) -> Tuple[List[float], float]:
        with torch.set_grad_enabled(True), torch.amp.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=(self.config.use_amp),
        ):
            (
                loss,
                preds,
                target,
                indexes,
                attn_scores,
                seq_len,
            ) = self.model(history, clicks, non_clicks, log=log)

            metrices = (
                [metric(preds, target, indexes=indexes) for metric in self.metrices]
                if log
                else []
            )

            self.optimizer.zero_grad()
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                self.optimizer.step()
            return metrices, loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, run, train: bool = True):
        if train and self.device_type == "cuda":
            dataloader.sampler.set_epoch(epoch)
        for iter, (history, clicks, non_clicks) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            history = history.to(self.local_rank)
            clicks = clicks.to(self.local_rank)
            non_clicks = non_clicks.to(self.local_rank)
            torch.cuda.empty_cache()
            metrices, batch_loss = (
                self._train_batch(history, clicks, non_clicks, iter % 100 == 0)
                if train
                else self._test_batch(history, clicks, non_clicks, iter % 100 == 0)
            )
            if train:
                self.scheduler.step()
            if iter % 100 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[RANK {self.local_rank}] Step{epoch}:{iter} | {step_type} Loss {batch_loss:.5f} |"
                    f" auc: {metrices[0]:.5f} | ndcg@5: {metrices[1]:.4f} | ndcg@10: {metrices[2]:.4f} | lr: {current_lr:.6f}"
                )
                if self.local_rank == 0 and run is not None:
                    if train:
                        run.log(
                            {
                                "train_loss": batch_loss,
                                "train_auc": metrices[0],
                                "train_ndcg_5": metrices[1],
                                "train_ndcg_10": metrices[2],
                                "learning": current_lr,
                            }
                        )
                    else:
                        run.log(
                            {
                                "eval_loss": batch_loss,
                                "eval_auc": metrices[0],
                                "eval_ndcg_5": metrices[1],
                                "eval_ndcg_10": metrices[2],
                            }
                        )

    def train(self, run=None):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            self._run_epoch(epoch, self.train_loader, run, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, run, train=False)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Parameter):
            # Initialize attention parameters with small values
            torch.nn.init.normal_(m, mean=0.0, std=0.02)

    def _prepare_for_metrics(self, user_repr, impressions, labels, samples_per_batch):
        relevance = (user_repr * impressions).sum(dim=-1)
        labels = (labels + 1) / 2
        indexes = torch.arange(
            self.config.batch_size, device=user_repr.device, dtype=torch.long
        )
        indexes = indexes.repeat_interleave(samples_per_batch, dim=0)
        return indexes, relevance, labels
