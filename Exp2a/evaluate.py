import argparse
from pathlib import Path
import yaml
import torch

from dataset import get_loaders
from models import build_model
from utils import get_device, pixel_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[
        "shallow_ae", "deep_ae", "sparse_ae", "vae", "beta_vae", "cvae"
    ])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = get_device(cfg)
    _, loader = get_loaders(cfg)

    model = build_model(args.model, cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mse = acc = count = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if args.model == "cvae":
                recon, _, _ = model(x, y)
            elif args.model in {"vae", "beta_vae"}:
                recon, _, _ = model(x)
            else:
                recon, _ = model(x)

            mse += torch.nn.functional.mse_loss(
                recon, x.flatten(1), reduction="sum"
            ).item()
            acc += pixel_accuracy(x, recon) * x.size(0)
            count += x.size(0)

    print(f"Model: {args.model}")
    print(f"Pixel reconstruction accuracy: {100 * acc / count:.2f}%")
    print(f"Mean squared error: {mse / count:.6f}")


if __name__ == "__main__":
    main()
