import torch.nn as nn


from .lif import LIFNode
from .izh import IzhikevichNode
from .hh import HHNode

def get_snn_node(snn_model):
    if not snn_model:
        return nn.Identity()
    if snn_model == "lif":
        return LIFNode()
    elif snn_model == "izh":
        return IzhikevichNode()
    elif snn_model == "hh":
        return HHNode()
    else:
        raise ValueError(f"Unknown SNN model: {snn_model}")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, snn_model="lif"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            get_snn_node(snn_model),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            get_snn_node(snn_model),
        )

    def forward(self, x):
        return self.block(x)
