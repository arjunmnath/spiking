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

    def _run_epoch(self, epoch: int, phase: str) -> Tuple[float, float]:
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

        total_loss = 0.0
        correct = 0
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

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

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
        accuracy = 100.0 * correct / total

        acc_tensor = torch.tensor(accuracy, device=self.device)
        acc_tensor = reduce_tensor(acc_tensor)
        if self.is_ddp:
            acc_tensor /= get_world_size()

        return avg_loss, acc_tensor.item()

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
            train_loss, train_acc = self._run_epoch(epoch, phase="train")

            if is_main_process():
                logger.info(f"--- Epoch {epoch} Summary ---")
                logger.info(
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
                )

            test_loss, test_acc = self._run_epoch(epoch, phase="eval")

            if is_main_process():
                logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
