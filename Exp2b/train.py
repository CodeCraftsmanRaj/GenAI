import argparse

from src.dataset import create_dataset
from src.models import build_models
from src.trainer import GANTrainer
from src.utils.helpers import (
    count_parameters,
    get_device,
    load_config,
    set_seed,
)


def main():

    # ====================================================
    # Arguments
    # ====================================================

    parser = argparse.ArgumentParser(
        description=(
            "Train GAN, cGAN or DCGAN "
            "on Fashion-MNIST"
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "gan",
            "cgan",
            "dcgan",
            "all"
        ],
        default=None,
    )

    parser.add_argument(
        "--config",
        default="config.yaml"
    )

    args = parser.parse_args()

    # ====================================================
    # Config
    # ====================================================

    config = load_config(
        args.config
    )

    set_seed(
        config["seed"]
    )

    # ====================================================
    # Device
    # ====================================================

    device = get_device()

    print(
        f"\nUsing device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{device}"
        )

    # ====================================================
    # Model selection
    # ====================================================

    selected_model = (
        args.model
        or config["models"]["run"]
    )

    if selected_model == "all":

        model_names = [
            "gan",
            "cgan",
            "dcgan"
        ]

    else:

        model_names = [
            selected_model
        ]

    # ====================================================
    # Dataset
    # ====================================================

    print(
        "\nLoading Fashion-MNIST..."
    )

    dataloader = create_dataset(
        config
    )

    print(
        f"Number of batches: "
        f"{len(dataloader)}"
    )

    # ====================================================
    # Train
    # ====================================================

    for model_name in model_names:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Training: "
            f"{model_name.upper()}"
        )

        print(
            "=" * 70
        )

        # -----------------------------------------------
        # Build models
        # -----------------------------------------------

        generator, discriminator = (
            build_models(
                model_name,
                config
            )
        )

        print(
            f"Generator parameters: "
            f"{count_parameters(generator):,}"
        )

        print(
            f"Discriminator parameters: "
            f"{count_parameters(discriminator):,}"
        )

        # -----------------------------------------------
        # Trainer
        # -----------------------------------------------

        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            model_name=model_name,
            config=config,
            device=device,
        )

        # -----------------------------------------------
        # Training
        # -----------------------------------------------

        trainer.fit(
            dataloader
        )


if __name__ == "__main__":
    main()