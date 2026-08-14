import random
from pathlib import Path
import numpy as np
import torch
from torchvision.utils import save_image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def reconstruction_loss(x, recon):
    x = x.flatten(1)
    return torch.nn.functional.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def pixel_accuracy(x, recon):
    x = x.flatten(1)
    pred = (recon >= 0.5).float()
    target = (x >= 0.5).float()
    return (pred == target).float().mean().item()


@torch.no_grad()
def save_reconstructions(model, loader, device, path, name):
    x, y = next(iter(loader))
    x = x.to(device)
    if name == "cvae":
        recon, _, _ = model(x, y.to(device))
    elif name in {"vae", "beta_vae"}:
        recon, _, _ = model(x)
    else:
        recon, _ = model(x)
    grid = torch.cat([x[:16], recon[:16].view(-1, 1, 28, 28)])
    save_image(grid.cpu(), Path(path) / f"{name}_recon.png", nrow=16)
