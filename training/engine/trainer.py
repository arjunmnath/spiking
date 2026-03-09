import os
import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.engine.checkpoint_manager import CheckpointManager
from training.utils.ddp import is_main_process, reduce_tensor, get_world_size, get_rank
import wandb
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision, ConfusionMatrix, Specificity, MatthewsCorrCoef

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    max_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    save_every: int = 5
    bucket_name: str = "my-cifar10-checkpoints"
    use_amp: bool = True
    grad_norm_clip: float = 1.0


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        device_type: str,
        is_ddp: bool,
    ):
        self.config = config
        self.device = device
        self.device_type = device_type
        self.is_ddp = is_ddp

        self.model = model.to(device)
        if self.is_ddp and self.device_type == "cuda":
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device.index]
            )

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()

        self.scaler = (
            torch.amp.GradScaler(device_type)
            if (config.use_amp and device_type in ["cuda"])
            else None
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs * len(train_loader)
        )

        self.epochs_run = 0

        metric_collection = MetricCollection({
            'acc': Accuracy(task="multiclass", num_classes=10),
            'acc_top5': Accuracy(task="multiclass", num_classes=10, top_k=5),
            'balanced_acc': Accuracy(task="multiclass", num_classes=10, average="macro"),
            'precision': Precision(task="multiclass", num_classes=10, average="macro"),
            'recall': Recall(task="multiclass", num_classes=10, average="macro"),
            'f1_macro': F1Score(task="multiclass", num_classes=10, average="macro"),
            'f1_micro': F1Score(task="multiclass", num_classes=10, average="micro"),
            'f1_weighted': F1Score(task="multiclass", num_classes=10, average="weighted"),
            'auroc': AUROC(task="multiclass", num_classes=10),
            'pr_auc': AveragePrecision(task="multiclass", num_classes=10),
            # ConfusionMatrix(task="multiclass", num_classes=10) # Logged manually due to non-scalar type
            'specificity': Specificity(task="multiclass", num_classes=10, average="macro"),
            'mcc': MatthewsCorrCoef(task="multiclass", num_classes=10),
        }).to(device)

        self.train_metrics = metric_collection.clone(prefix="train_")
        self.val_metrics = metric_collection.clone(prefix="val_")

    def _run_epoch(self, epoch: int, phase: str) -> Tuple[float, dict]:
        assert phase in ["train", "eval"]
        is_train = phase == "train"

        if is_train:
            self.model.train()
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            dataloader = self.train_loader
        else:
            self.model.eval()
            dataloader = self.test_loader

        metrics = self.train_metrics if is_train else self.val_metrics
        metrics.reset()

        total_loss = 0.0
        total = 0

        with torch.set_grad_enabled(is_train):
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                self.model.apply(lambda m: getattr(m, 'reset', lambda: None)())
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                if self.scaler is not None:
                    with torch.amp.autocast(
                        device_type=self.device_type, dtype=torch.float16
                    ):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    if is_train:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_norm_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    if is_train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_norm_clip
                        )
                        self.optimizer.step()

                if is_train:
                    self.scheduler.step()

                total += labels.size(0)
                metrics.update(outputs, labels)

                reduced_loss = reduce_tensor(loss.detach())
                if self.is_ddp:
                    reduced_loss /= get_world_size()
                total_loss += reduced_loss.item()

                if is_train and batch_idx % 100 == 0 and is_main_process():
                    curr_lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch} | Step {batch_idx}/{len(dataloader)} | "
                        f"Train Loss: {reduced_loss.item():.4f} | LR: {curr_lr:.6f}"
                    )

        avg_loss = total_loss / len(dataloader)
        metrics_res = metrics.compute()

        return avg_loss, metrics_res

    def save_checkpoint(self, epoch: int):
        if self.config.bucket_name:
            with CheckpointManager(self.config.bucket_name) as manager:
                model_state = (
                    self.model.module.state_dict()
                    if self.is_ddp and self.device_type == "cuda"
                    else self.model.state_dict()
                )
                optim_state = self.optimizer.state_dict()
                meta_data = {
                    "epoch": epoch,
                    "config": self.config.__dict__,
                }
                manager.save_checkpoint(
                    model_data=model_state,
                    optimizer_data=optim_state,
                    meta_data=meta_data,
                    step=epoch,
                    rank=get_rank(),
                )
        elif is_main_process():
            os.makedirs("checkpoints", exist_ok=True)
            model_state = (
                self.model.module.state_dict()
                if self.is_ddp and self.device_type == "cuda"
                else self.model.state_dict()
            )
            torch.save(
                {
                    "model_state_dict": model_state,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                f"checkpoints/epoch_{epoch}.pt",
            )

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            train_loss, train_metrics = self._run_epoch(epoch, phase="train")

            if is_main_process():
                logger.info(f"--- Epoch {epoch} Summary ---")
                logger.info(
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['train_acc'].item() * 100:.2f}%"
                )

            val_loss, val_metrics = self._run_epoch(epoch, phase="eval")

            if is_main_process():
                logger.info(f"Test Loss: {val_loss:.4f} | Test Acc: {val_metrics['val_acc'].item() * 100:.2f}%")
                
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                
                for k, v in train_metrics.items():
                    log_dict[k] = v.item()
                for k, v in val_metrics.items():
                    log_dict[k] = v.item()
                    
                if wandb.run is not None:
                    wandb.log(log_dict)

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
