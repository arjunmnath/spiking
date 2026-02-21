import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
# from utils.ddp import is_main_process, reduce_tensor
from training.utils.ddp import reduce_tensor
import torch.distributed as dist

class Trainer:
    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Determine device_ids for DDP
        if device.type == "cuda":
            model = model.to(device)
            self.model = DDP(model, device_ids=[device.index])
        elif device.type == "mps":
            model = model.to(device)
            self.model = model # DDP not natively supported on MPS the same way as CUDA
        else:
            model = model.to(device)
            self.model = model # CPU fallback
            
        self.scaler = GradScaler(device.type) if device.type == "cuda" else None
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.model.train()
        
        # DistributedSampler requires setting the epoch for shuffling
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
            
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.scaler is not None:
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Reduce loss to rank 0 for logging
            reduced_loss = reduce_tensor(loss.detach()) 
            if dist.is_initialized():
                reduced_loss /= dist.get_world_size()

            total_loss += reduced_loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Ensure correct scaling of accuracy across GPUs
        accuracy_tensor = torch.tensor(accuracy, device=self.device)
        accuracy_tensor = reduce_tensor(accuracy_tensor)
        if dist.is_initialized():
             accuracy_tensor /= dist.get_world_size()

        return avg_loss, accuracy_tensor.item()

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.scaler is not None:
                    with autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                reduced_loss = reduce_tensor(loss.detach())
                if dist.is_initialized():
                    reduced_loss /= dist.get_world_size()

                total_loss += reduced_loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        accuracy_tensor = torch.tensor(accuracy, device=self.device)
        accuracy_tensor = reduce_tensor(accuracy_tensor)
        if dist.is_initialized():
             accuracy_tensor /= dist.get_world_size()

        return avg_loss, accuracy_tensor.item()

    def save_checkpoint(self, path):
        if is_main_process():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
