from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return yaml.safe_load(file)


def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_image_grid(
    images,
    path,
    title=None,
    nrow=10
):

    images = images.detach().cpu()

    # [-1,1] -> [0,1]
    images = (images + 1) / 2

    images = torch.clamp(
        images,
        0,
        1
    )

    images = images.numpy()

    num_images = images.shape[0]

    ncols = nrow

    nrows = (
        num_images + ncols - 1
    ) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 10)
    )

    axes = np.array(
        axes
    ).reshape(
        nrows,
        ncols
    )

    for i, ax in enumerate(
        axes.flat
    ):

        if i < num_images:

            ax.imshow(
                images[i].squeeze(),
                cmap="gray"
            )

        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_checkpoint(
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    epoch,
    path
):

    checkpoint = {

        "epoch": epoch,

        "generator": generator.state_dict(),

        "discriminator":
            discriminator.state_dict(),

        "generator_optimizer":
            g_optimizer.state_dict(),

        "discriminator_optimizer":
            d_optimizer.state_dict(),
    }

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        checkpoint,
        path
    )


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )