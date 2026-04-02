import torchvision
from torchvision import transforms

def get_transforms():
    transform_original = transforms.Compose([
        transforms.ToTensor(),  # (3, 32, 32)
    ])

    transform_compressed = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # (1, 16, 16)
    ])

    return transform_original, transform_compressed

t_orig, t_comp = get_transforms()

dataset_orig = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=t_orig
)

dataset_comp = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=t_comp
)


import matplotlib.pyplot as plt

classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def plot_comparison(dataset_orig, dataset_comp, n=5):
    plt.figure(figsize=(6, 2*n))

    for i in range(n):
        img_orig, label = dataset_orig[i]
        img_comp, _ = dataset_comp[i]

        img_orig = img_orig.permute(1, 2, 0)  # (H, W, C)
        img_comp = img_comp.squeeze(0)        # (16,16)

        # Original
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(img_orig)
        plt.title(f"Original: {classes[label]}")
        plt.axis('off')

        # Compressed
        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(img_comp, cmap='gray')
        plt.title(f"Compressed: {classes[label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

plot_comparison(dataset_orig, dataset_comp, n=6)