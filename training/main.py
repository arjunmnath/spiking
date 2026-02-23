import logging
import os
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG

from config_classes import DataConfig, OptimizerConfig, TrainingConfig
from dataset import Mind
from models import *
from models.models import TwoTowerRecommendation
from trainer import Trainer

from utils import create_optimizer

logger = logging.getLogger(__name__)

def verify_min_gpu_count(min_gpus: int = 1) -> bool:
    """
    Verifies if there are enough GPUs available for training.

    Args:
        min_gpus (int): Minimum number of GPUs required.

    Returns:
        bool: True if the required number of GPUs is available, False otherwise.
    """
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus


def ddp_setup():
    """
    Sets up Distributed Data Parallel (DDP) for multi-GPU training.

    This function initializes the process group and sets the device for each process.
    """
    acc = torch.accelerator.current_accelerator()
    rank = int(os.environ["LOCAL_RANK"])
    device: torch.device = torch.device(f"{acc}:{rank}")
    backend = torch.distributed.get_default_backend_for_device(device)
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(rank)
    return device


def get_train_objs(data_cfg: DataConfig, opt_cfg: OptimizerConfig):
    """
    Initializes training objects including datasets, model, loss function, and metrics.

    Args:
        data_cfg (DataConfig): Configuration object containing data-related settings.
        opt_cfg (OptimizerConfig): Configuration object containing optimizer-related settings.

    Returns:
        tuple: Contains the model, optimizer, loss function, metrics, training dataset, and testing dataset.
    """
    data_dir = Path(data_cfg.data_dir)
    embed_dir = Path(data_cfg.embed_dir)
    train_dataset = Mind(
        dataset_dir=data_dir / "train",
        precompute=data_cfg.precompute,
        embed_dir=embed_dir / "train",
    )
    test_dataset = Mind(
        dataset_dir=data_dir / "test",
        precompute=data_cfg.precompute,
        embed_dir=embed_dir / "test",
    )
    auc_roc = RetrievalAUROC()
    ndcg_5 = RetrievalNormalizedDCG(top_k=5)
    ndcg_10 = RetrievalNormalizedDCG(top_k=10)
    model = TwoTowerRecommendation()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg.learning_rate, betas=(0.9, 0.99)
    )
    return (
        model,
        optimizer,
        [auc_roc, ndcg_5, ndcg_10],
        train_dataset,
        test_dataset,
    )


@hydra.main(version_base=None, config_path=".", config_name="mind_train_cfg.yaml")
def main(cfg: DictConfig):
    # Setup logging
    from logging_config import setup_logging

    setup_logging()

    device = ddp_setup()

    # configs
    opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
    data_cfg = DataConfig(**cfg["data_config"])
    trainer_cfg = TrainingConfig(**cfg["trainer_config"])

    rank = int(os.environ["LOCAL_RANK"])
    model, optimizer, metrices, train_data, test_data = get_train_objs(
        data_cfg, opt_cfg
    )
    trainer = Trainer(
        config=trainer_cfg,
        model=model,
        metrices=metrices,
        optimizer=optimizer,
        train_dataset=train_data,
        test_dataset=test_data,
    )
    if rank == 0:
        with wandb.init(project="mind", entity="arjunmnath-iiitkottayam") as run:
            run.config.batch_size = trainer_cfg.batch_size
            run.config.learning_rate = opt_cfg.learning_rate
            run.config.weight_decay = opt_cfg.weight_decay
            run.config.epochs = trainer_cfg.max_epochs
            run.config.optimizer = "AdamW"
            run.config.metrics = ([metric.__class__.__name__ for metric in metrices],)
            run.config.loss_function = "InfoNCE"
            trainer.train(run)
    else:
        trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    _min_gpu_count = 1
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        logger.error(
            f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting."
        )
        sys.exit()
    main()
