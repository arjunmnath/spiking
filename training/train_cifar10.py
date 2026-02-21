import os
import yaml
import torch

from data.cifar10_dataset import CIFAR10DataModule
from models import ImageClassifier
from engine.cifar10_trainer import Trainer
from utils.ddp import init_distributed, cleanup, is_main_process

def load_config(path="configs/cifar10.yaml"):
    if not os.path.exists(path):
        return {"training": {"epochs": 10, "batch_size": 128, "learning_rate": 0.001, "num_workers": 4, "log_every_n_steps": 10}}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Setup Distributed Environment
    is_distributed = init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    if is_main_process():
        print(f"Using device: {device}")
        if is_distributed:
            print(f"DDP Initialized with World Size: {os.environ.get('WORLD_SIZE')}")
    
    # Load config
    config = load_config()
    train_cfg = config["training"]
    
    # 2. Setup Data
    data_module = CIFAR10DataModule(
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"]
    )
    train_loader, test_loader = data_module.get_dataloaders()
    
    # 3. Initialize Model & Optimizer
    model = SimpleCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    
    # 4. Initialize Trainer
    trainer = Trainer(model, optimizer, train_loader, test_loader, device)
    
    # 5. Training Loop
    epochs = train_cfg["epochs"]
    for epoch in range(epochs):
        if is_main_process():
            print(f"-- Epoch {epoch+1}/{epochs} --")
            
        train_loss, train_acc = trainer.train_epoch(epoch)
        
        # Evaluate at the end of every epoch
        val_loss, val_acc = trainer.evaluate()
        
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            trainer.save_checkpoint(f"checkpoints/epoch_{epoch+1}.pt")
            
    # 6. Tear down distributed environment
    cleanup()

if __name__ == "__main__":
    main()
