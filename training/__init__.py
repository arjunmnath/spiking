from training.data.dataset import CIFAR10
import torch
import torch.nn.functional as F
from training.models import ImageClassifier



dataset = CIFAR10(batch_size=8)
train_loader, test_loader = dataset.get_dataloaders()
model = ImageClassifier()

for img, label in train_loader:
    a = F.softmax(model(img), dim=1)
    a = torch.argmax(a, dim=1).to(torch.float)
    print(a.mean(), a.std())
    break
