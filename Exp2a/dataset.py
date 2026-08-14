import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(cfg):
    tfm = transforms.ToTensor()
    root = cfg["data"]["data_dir"]
    bs = cfg["data"]["batch_size"]

    train_ds = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root, train=False, download=True, transform=tfm)

    kw = {
        "batch_size": bs,
        "num_workers": cfg["data"]["num_workers"],
        "pin_memory": cfg["data"]["pin_memory"],
    }
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(test_ds, shuffle=False, **kw)
