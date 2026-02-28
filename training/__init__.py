import dataclasses

from training.engine.checkpoint_manager import CheckpointManager
from training.utils.common import get_run_id

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

@dataclass
class ModelConfig:
    def __init__(self, input_size=10, hidden_size=50, output_size=2, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

    def __repr__(self):
        return f"ModelConfig(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, lr={self.lr})"

class SimpleModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.output_size)
        self.config = config  # Store the configuration for later use

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


config = ModelConfig(input_size=10, hidden_size=50, output_size=2, lr=0.001)
model = SimpleModel(config)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example metadata (you could have more detailed config)
metadata = {
    "model_type": "SimpleModel",
    "optimizer": "Adam",
    "learning_rate": config.lr,
    "model_config": config.__dict__,
}


with CheckpointManager(bucket_name='ckpt-spikign') as manager:
    manager.save_checkpoint(model, optimizer, metadata, step=1)

# run_id = "neatly-native-shark"
# manager = CheckpointManager(bucket_name='ckpt-spikign')
# model, metadata = manager.build_model_from_run_id(SimpleModel, ModelConfig, run_id, torch.device("meta"), "train", dirty_load=True)
# print(model, metadata)
