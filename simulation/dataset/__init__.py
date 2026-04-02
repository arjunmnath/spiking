from torchvision import transforms
import torchvision
import numpy as np

def get_numpy_cifar2_splits(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((16, 16)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    def process(split):
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=(split == "train"),
            download=True
        )

        data = dataset.data
        targets = np.array(dataset.targets)

        mask = (targets == 0) | (targets == 6)
        data = data[mask]
        targets = targets[mask]
        targets = (targets == 6).astype(np.int64)

        X = np.stack([transform(img).numpy() for img in data])
        y = targets
        return X, y

    X_train, y_train = process("train")
    X_test, y_test = process("test")
    return X_train.squeeze(1), y_train, X_test.squeeze(1), y_test
