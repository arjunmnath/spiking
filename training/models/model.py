import torch.nn as nn
from .blocks import ConvBlock
from .lif import LIFNode
from .izh import IzhikevichNode
from .hh import HHNode

def get_snn_node(snn_model):
    if snn_model == "lif":
        return LIFNode()
    elif snn_model == "izh":
        return IzhikevichNode()
    elif snn_model == "hh":
        return HHNode()
    else:
        raise ValueError(f"Unknown SNN model: {snn_model}")


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5, snn_model="lif"):
        super().__init__()

        self.stage1 = nn.Sequential(ConvBlock(3, 64, snn_model), nn.MaxPool2d(2))  # 32 -> 16

        self.stage2 = nn.Sequential(ConvBlock(64, 128, snn_model), nn.MaxPool2d(2))  # 16 -> 8

        self.stage3 = nn.Sequential(ConvBlock(128, 256, snn_model), nn.MaxPool2d(2))  # 8 -> 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            get_snn_node(snn_model),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x
