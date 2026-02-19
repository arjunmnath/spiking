import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.utils.ddp import is_initialized
from training.data.transforms import get_transforms

class CIFAR10:
    def __init__(self, data_dir="./data", batch_size=128, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train, self.transform_test = get_transforms()

    def get_dataloaders(self):
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.transform_test
        )
        train_sampler = DistributedSampler(trainset) if is_initialized() else None
        trainloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            sampler=train_sampler,
            pin_memory=True
        )

        testloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return trainloader, testloader