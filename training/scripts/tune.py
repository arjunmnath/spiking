import json
import os
import argparse
import random
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import wandb
from optuna.integration.pytorch_distributed import TorchDistributedTrial
from torch.nn.parallel import DistributedDataParallel as DDP

from training.data.dataset import CIFAR10
from training.models.model import ImageClassifier
from training.utils.common import get_run_id
from training.utils.ddp import (
    compute_init,
    cleanup,
    is_main_process,
    get_world_size,
    get_rank,
    reduce_tensor,
    is_initialized,
)

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Ensure deterministic behavior."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(trial, args):
    """Instantiate model with tunable hyperparameters."""
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    model = ImageClassifier(num_classes=10, dropout_rate=dropout_rate, snn_model=args.snn_model)
    return model


def build_optimizer(trial, model):
    """Instantiate optimizer with tunable hyperparameters."""
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    return optimizer


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, scaler, grad_clip, is_ddp
):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        model.apply(lambda m: getattr(m, 'reset', lambda: None)())
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if is_ddp:
        t = torch.tensor(
            [total_loss, correct, total], device=device, dtype=torch.float64
        )
        t = reduce_tensor(t)
        avg_loss = (t[0] / t[2]).item()
        accuracy = 100.0 * (t[1] / t[2]).item()
    else:
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, scaler, is_ddp):
    """Evaluate the model on the validation dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            model.apply(lambda m: getattr(m, 'reset', lambda: None)())
            inputs, labels = inputs.to(device), labels.to(device)

            if scaler is not None:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    if is_ddp:
        t = torch.tensor(
            [total_loss, correct, total], device=device, dtype=torch.float64
        )
        t = reduce_tensor(t)
        avg_loss = (t[0] / t[2]).item()
        accuracy = 100.0 * (t[1] / t[2]).item()
    else:
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def objective(trial, args, device, is_ddp):
    """Optuna objective function for tuning."""
    if is_ddp:
        trial = TorchDistributedTrial(trial)

    if is_main_process():
        base_run_name = args.run_name if args.run_name else get_run_id()
        run_name = f"{base_run_name}_trial_{trial.number}"
        wandb.init(
            project="cifar10",
            group="optuna-search",
            name=run_name,
            config=trial.params,
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )

    set_seed(42 + trial.number)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    dataset = CIFAR10(batch_size=batch_size, num_workers=args.num_workers)
    full_train_loader, _ = dataset.get_dataloaders()
    full_train_dataset = full_train_loader.dataset

    val_size = int(len(full_train_dataset) * args.val_split_ratio)
    train_size = len(full_train_dataset) - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42 + trial.number)
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset) if is_ddp else None
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False) if is_ddp else None
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=True
    )

    model = build_model(trial, args).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if is_main_process():
        wandb.watch(model)
    optimizer = build_optimizer(trial, model if not is_ddp else model.module)
    criterion = nn.CrossEntropyLoss()

    scaler = (
        torch.amp.GradScaler(device.type)
        if args.use_amp and device.type in ["cuda"]
        else None
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        if is_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.grad_clip,
            is_ddp,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, scaler, is_ddp
        )

        if is_main_process():

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

        trial.report(val_loss, epoch)

        if trial.should_prune():
            if is_main_process():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                os.makedirs("tuning_artifacts", exist_ok=True)
                data = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "trial_params": trial.params,
                }
                json_path = "tuning_artifacts/best_trial.json"
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                artifact = wandb.Artifact("best_tuning_trial", type="tuning")
                artifact.add_file(json_path)
                wandb.log_artifact(artifact)

    if is_main_process():
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs per trial"
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of optuna trials"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.2,
        help="Ratio of the train dataset to use for validation",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name for wandb. If not provided, a random ID will be generated.",
    )
    parser.add_argument(
        "--snn_model",
        type=str,
        choices=["lif", "izh", "hh"],
        default="lif",
        help="Target SNN model to use (lif, izh, hh)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 epoch and 1 trial for testing",
    )
    args = parser.parse_args()

    if args.fast_dev_run:
        args.epochs = 1
        args.trials = 1
        args.device = "cpu"
        args.num_workers = 0

    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(
        args.device
    )
    is_ddp = is_initialized()

    if is_main_process():
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Starting tuning on device: {device}, is_ddp: {is_ddp}")

    study = None
    if is_main_process():
        study = optuna.create_study(
            direction="minimize",
            study_name="cifar10_tuning",
            pruner=optuna.pruners.MedianPruner(),
        )

    try:
        if is_main_process():
            study.optimize(
                lambda trial: objective(trial, args, device, is_ddp),
                n_trials=args.trials,
            )
        else:
            for _ in range(args.trials):
                try:
                    objective(None, args, device, is_ddp)
                except optuna.exceptions.TrialPruned:
                    pass
    except KeyboardInterrupt:
        if is_main_process():
            logger.info("Tuning interrupted by user.")

    if is_main_process() and study is not None:
        logger.info("Tuning finished.")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"  Value (Validation Loss): {study.best_trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

    cleanup()


if __name__ == "__main__":
    main()
