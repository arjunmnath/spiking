import logging
from torch.optim import AdamW

from training.data.dataset import CIFAR10
from training.models.model import ImageClassifier
from training.engine.trainer import Trainer, TrainingConfig
from training.utils.ddp import compute_init, cleanup, is_main_process

logger = logging.getLogger(__name__)


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
    optimizer = AdamW(
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
