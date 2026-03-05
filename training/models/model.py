import torch.nn as nn
from .blocks import ConvBlock


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()

        self.stage1 = nn.Sequential(ConvBlock(3, 64), nn.MaxPool2d(2))  # 32 -> 16

        self.stage2 = nn.Sequential(ConvBlock(64, 128), nn.MaxPool2d(2))  # 16 -> 8

        self.stage3 = nn.Sequential(ConvBlock(128, 256), nn.MaxPool2d(2))  # 8 -> 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x
