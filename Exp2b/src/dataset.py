from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_transform():
    """
    Fashion-MNIST:

        [0, 255]
            ↓
        [0, 1]
            ↓
        [-1, 1]

    The [-1, 1] range matches the Generator's Tanh output.
    """

    return transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])


def create_dataset(config):
    """
    Download/load Fashion-MNIST and create DataLoader.
    """

    data_dir = Path(
        config["dataset"]["data_dir"]
    )

    batch_size = config["dataset"]["batch_size"]

    num_workers = config["dataset"]["num_workers"]

    transform = get_transform()

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return dataloader