import os
import argparse
from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Tuple

from training.data.dataset import CIFAR10
from training.models.model import ImageClassifier
from training.engine.checkpoint_manager import CheckpointManager
from training.utils.ddp import (
    compute_init,
    cleanup,
    is_main_process,
    reduce_tensor,
    get_world_size,
    get_rank
)

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
            is_ddp: bool
    ):
        self.config = config
        self.device = device
        self.device_type = device_type
        self.is_ddp = is_ddp

        self.model = model.to(device)
        if self.is_ddp and self.device_type == "cuda":
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[device.index])

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()

        # AMP Scaler
        self.scaler = torch.amp.GradScaler(device_type) if (config.use_amp and device_type in ["cuda"]) else None

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs * len(train_loader)
        )

        self.epochs_run = 0

    def _train_epoch(self, epoch: int):
        self.model.train()

        # Set epoch for DistributedSampler
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast(device_type=self.device_type, dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

            self.scheduler.step()

            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Reduce loss across devices for accurate logging
            reduced_loss = reduce_tensor(loss.detach())
            if self.is_ddp:
                reduced_loss /= get_world_size()
            total_loss += reduced_loss.item()

            if batch_idx % 100 == 0 and is_main_process():
                curr_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch} | Step {batch_idx}/{len(self.train_loader)} | "
                            f"Loss: {reduced_loss.item():.4f} | LR: {curr_lr:.6f}")

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        # Sync accuracy
        acc_tensor = torch.tensor(accuracy, device=self.device)
        acc_tensor = reduce_tensor(acc_tensor)
        if self.is_ddp:
            acc_tensor /= get_world_size()

        return avg_loss, acc_tensor.item()

    def _test_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.scaler is not None:
                    with torch.amp.autocast(device_type=self.device_type, dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                reduced_loss = reduce_tensor(loss.detach())
                if self.is_ddp:
                    reduced_loss /= get_world_size()
                total_loss += reduced_loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        # Sync accuracy
        acc_tensor = torch.tensor(accuracy, device=self.device)
        acc_tensor = reduce_tensor(acc_tensor)
        if self.is_ddp:
            acc_tensor /= get_world_size()

        return avg_loss, acc_tensor.item()

    def save_checkpoint(self, epoch: int):
        if self.config.bucket_name:
            with CheckpointManager(self.config.bucket_name) as manager:
                model_state = self.model.module.state_dict() if self.is_ddp and self.device_type == "cuda" else self.model.state_dict()
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
                    rank=get_rank()
                )
        elif is_main_process():
            # Local fallback
            os.makedirs("checkpoints", exist_ok=True)
            model_state = self.model.module.state_dict() if self.is_ddp and self.device_type == "cuda" else self.model.state_dict()
            torch.save({
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, f"checkpoints/epoch_{epoch}.pt")

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            train_loss, train_acc = self._train_epoch(epoch)

            if is_main_process():
                logger.info(f"--- Epoch {epoch} Summary ---")
                logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

            test_loss, test_acc = self._test_epoch(epoch)

            if is_main_process():
                logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)


def main():
    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig()

    # Initialize DDP / Device
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = device.type

    if is_main_process():
        logger.info(f"Initialized with device: {device}, is_ddp: {is_ddp_requested}")

    # Setup data
    dataset = CIFAR10(batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader, test_loader = dataset.get_dataloaders()

    # Initialize model and optimizer
    model = ImageClassifier(num_classes=10)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        device_type=device_type,
        is_ddp=is_ddp_requested
    )

    if is_main_process():
        logger.info("Starting training...")

    trainer.train()

    cleanup()


if __name__ == "__main__":
    main()
