import argparse
from pathlib import Path
import yaml
import torch
from tqdm import tqdm

from dataset import get_loaders
from models import build_model
from utils import seed_everything, get_device, reconstruction_loss, kl_loss, pixel_accuracy, save_reconstructions


def run_epoch(model, loader, opt, device, name, cfg, train=True):
    model.train(train)
    total = rec_total = kl_total = sparse_total = acc_total = 0.0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()

        if name == "cvae":
            recon, mu, logvar = model(x, y)
        elif name in {"vae", "beta_vae"}:
            recon, mu, logvar = model(x)
        else:
            recon, z = model(x)

        rec = reconstruction_loss(x, recon)
        kl = kl_loss(mu, logvar) if name in {"vae", "beta_vae", "cvae"} else x.new_tensor(0.)
        sparse = z.abs().mean() if name == "sparse_ae" else x.new_tensor(0.)
        beta = cfg["training"]["beta"] if name == "beta_vae" else 1.0
        loss = rec + beta * kl + cfg["training"]["sparsity_lambda"] * sparse

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            opt.step()

        n = x.size(0)
        total += loss.item() * n
        rec_total += rec.item() * n
        kl_total += kl.item() * n
        sparse_total += sparse.item() * n
        acc_total += pixel_accuracy(x, recon) * n

    n = len(loader.dataset)
    return {
        "loss": total / n,
        "recon": rec_total / n,
        "kl": kl_total / n,
        "sparse": sparse_total / n,
        "pixel_acc": acc_total / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[
        "shallow_ae", "deep_ae", "sparse_ae", "vae", "beta_vae", "cvae"
    ])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])
    device = get_device(cfg)
    out = Path(cfg["experiment"]["output_dir"]) / args.model
    out.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(cfg)
    model = build_model(args.model, cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    best = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr = run_epoch(model, train_loader, opt, device, args.model, cfg, True)
        te = run_epoch(model, test_loader, opt, device, args.model, cfg, False)

        print(
            f"Epoch {epoch:02d} | "
            f"train {tr['loss']:.4f} | test {te['loss']:.4f} | "
            f"recon {te['recon']:.4f} | pixel_acc {te['pixel_acc']:.4f}"
        )

        if te["loss"] < best:
            best = te["loss"]
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "model_name": args.model,
                "metrics": te,
            }, out / "best.pt")

    save_reconstructions(model, test_loader, device, out, args.model)
    print(f"\nSaved: {out / 'best.pt'}")
    print(f"Final pixel reconstruction accuracy: {te['pixel_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
