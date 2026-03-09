import argparse
import logging
import torch.optim as optim
from torch.optim import AdamW

from training.data.dataset import CIFAR10
from training.models.model import ImageClassifier
from training.engine.trainer import Trainer, TrainingConfig
from training.utils.ddp import compute_init, cleanup, is_main_process

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snn_model", type=str, choices=["lif", "izh", "hh"], default="lif")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW"], default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bucket_name", type=str, default="", help="S3 bucket name for checkpoints")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        bucket_name=args.bucket_name,
    )

    # Initialize DDP / Device
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = device.type

    if is_main_process():
        logger.info(f"Initialized with device: {device}, is_ddp: {is_ddp_requested}")

    # Setup data
    dataset = CIFAR10(batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader, test_loader = dataset.get_dataloaders()

    # Initialize model and optimizer
    model = ImageClassifier(num_classes=10, dropout_rate=args.dropout, snn_model=args.snn_model)
    if args.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        device_type=device_type,
        is_ddp=is_ddp_requested,
    )

    if is_main_process():
        logger.info("Starting training...")

    trainer.train()

    cleanup()


if __name__ == "__main__":
    main()
