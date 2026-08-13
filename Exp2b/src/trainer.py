from pathlib import Path

import torch
import torch.nn as nn

from src.utils.helpers import (
    save_checkpoint,
    save_image_grid,
)


class GANTrainer:

    def __init__(
        self,
        generator,
        discriminator,
        model_name,
        config,
        device,
    ):

        self.generator = generator.to(device)

        self.discriminator = (
            discriminator.to(device)
        )

        self.model_name = model_name

        self.config = config

        self.device = device

        self.latent_dim = (
            config["training"]["latent_dim"]
        )

        self.epochs = (
            config["training"]["epochs"]
        )

        self.num_classes = (
            config["dataset"]["num_classes"]
        )

        # ------------------------------------------------
        # Loss
        # ------------------------------------------------

        self.criterion = (
            nn.BCEWithLogitsLoss()
        )

        # ------------------------------------------------
        # Optimizers
        # ------------------------------------------------

        generator_lr = (
            config["training"]["generator_lr"]
        )

        discriminator_lr = (
            config["training"]["discriminator_lr"]
        )

        beta1 = (
            config["training"]["beta1"]
        )

        beta2 = (
            config["training"]["beta2"]
        )

        self.g_optimizer = (
            torch.optim.Adam(
                self.generator.parameters(),
                lr=generator_lr,
                betas=(beta1, beta2),
            )
        )

        self.d_optimizer = (
            torch.optim.Adam(
                self.discriminator.parameters(),
                lr=discriminator_lr,
                betas=(beta1, beta2),
            )
        )

        # ------------------------------------------------
        # Output directories
        # ------------------------------------------------

        output_dir = Path(
            config["training"]["output_dir"]
        ) / model_name

        self.sample_dir = (
            output_dir / "samples"
        )

        self.checkpoint_dir = (
            output_dir / "checkpoints"
        )

        self.sample_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ------------------------------------------------
        # Fixed noise
        # ------------------------------------------------

        self.fixed_noise = torch.randn(
            100,
            self.latent_dim,
            device=self.device
        )

        # cGAN gets one fixed label per generated image
        if self.model_name == "cgan":

            self.fixed_labels = (
                torch.arange(
                    self.num_classes,
                    device=self.device
                ).repeat_interleave(10)
            )

            self.fixed_noise = torch.randn(
                len(self.fixed_labels),
                self.latent_dim,
                device=self.device
            )

    # ====================================================
    # GENERATOR
    # ====================================================

    def generate(
        self,
        noise,
        labels=None
    ):

        if self.model_name == "cgan":

            return self.generator(
                noise,
                labels
            )

        return self.generator(
            noise
        )

    # ====================================================
    # DISCRIMINATOR
    # ====================================================

    def discriminate(
        self,
        images,
        labels=None
    ):

        if self.model_name == "cgan":

            return self.discriminator(
                images,
                labels
            )

        return self.discriminator(
            images
        )

    # ====================================================
    # TRAIN DISCRIMINATOR
    # ====================================================

    def train_discriminator(
        self,
        real_images,
        labels
    ):

        batch_size = (
            real_images.size(0)
        )

        noise = torch.randn(
            batch_size,
            self.latent_dim,
            device=self.device
        )

        # -----------------------------------------------
        # Generate fake images
        # -----------------------------------------------

        with torch.no_grad():

            if self.model_name == "cgan":

                fake_images = self.generate(
                    noise,
                    labels
                )

            else:

                fake_images = self.generate(
                    noise
                )

        # -----------------------------------------------
        # Real
        # -----------------------------------------------

        real_output = self.discriminate(
            real_images,
            labels
        )

        real_targets = torch.full_like(
            real_output,
            0.9
        )

        real_loss = self.criterion(
            real_output,
            real_targets
        )

        # -----------------------------------------------
        # Fake
        # -----------------------------------------------

        fake_output = self.discriminate(
            fake_images,
            labels
        )

        fake_targets = torch.zeros_like(
            fake_output
        )

        fake_loss = self.criterion(
            fake_output,
            fake_targets
        )

        # -----------------------------------------------
        # Total discriminator loss
        # -----------------------------------------------

        d_loss = (
            real_loss + fake_loss
        ) / 2

        self.d_optimizer.zero_grad()

        d_loss.backward()

        self.d_optimizer.step()

        return d_loss.item()

    # ====================================================
    # TRAIN GENERATOR
    # ====================================================

    def train_generator(
        self,
        labels
    ):

        batch_size = labels.size(0)

        noise = torch.randn(
            batch_size,
            self.latent_dim,
            device=self.device
        )

        if self.model_name == "cgan":

            fake_images = self.generate(
                noise,
                labels
            )

            fake_output = self.discriminate(
                fake_images,
                labels
            )

        else:

            fake_images = self.generate(
                noise
            )

            fake_output = self.discriminate(
                fake_images
            )

        # Generator wants discriminator
        # to classify fake images as real

        targets = torch.ones_like(
            fake_output
        )

        g_loss = self.criterion(
            fake_output,
            targets
        )

        self.g_optimizer.zero_grad()

        g_loss.backward()

        self.g_optimizer.step()

        return g_loss.item()

    # ====================================================
    # SAVE SAMPLES
    # ====================================================

    @torch.no_grad()
    def save_samples(
        self,
        epoch
    ):

        self.generator.eval()

        if self.model_name == "cgan":

            images = self.generate(
                self.fixed_noise,
                self.fixed_labels
            )

        else:

            images = self.generate(
                self.fixed_noise
            )

        path = (
            self.sample_dir
            / f"epoch_{epoch:04d}.png"
        )

        save_image_grid(
            images,
            path,
            title=(
                f"{self.model_name.upper()} "
                f"- Epoch {epoch}"
            ),
            nrow=10,
        )

        self.generator.train()

    # ====================================================
    # TRAIN
    # ====================================================

    def fit(self, dataloader):

        sample_every = (
            self.config["training"]
            ["sample_every"]
        )

        checkpoint_every = (
            self.config["training"]
            ["checkpoint_every"]
        )

        for epoch in range(
            1,
            self.epochs + 1
        ):

            d_losses = []

            g_losses = []

            for real_images, labels in dataloader:

                real_images = (
                    real_images.to(
                        self.device,
                        non_blocking=True
                    )
                )

                labels = (
                    labels.to(
                        self.device,
                        non_blocking=True
                    )
                )

                # Train D
                d_loss = (
                    self.train_discriminator(
                        real_images,
                        labels
                    )
                )

                # Train G
                g_loss = (
                    self.train_generator(
                        labels
                    )
                )

                d_losses.append(
                    d_loss
                )

                g_losses.append(
                    g_loss
                )

            mean_d_loss = (
                sum(d_losses)
                / len(d_losses)
            )

            mean_g_loss = (
                sum(g_losses)
                / len(g_losses)
            )

            print(
                f"Epoch "
                f"{epoch:03d}/{self.epochs} | "
                f"D Loss: "
                f"{mean_d_loss:.4f} | "
                f"G Loss: "
                f"{mean_g_loss:.4f}"
            )

            # -------------------------------------------
            # Samples
            # -------------------------------------------

            if (
                epoch % sample_every == 0
                or epoch == 1
            ):

                self.save_samples(
                    epoch
                )

            # -------------------------------------------
            # Checkpoint
            # -------------------------------------------

            if (
                epoch % checkpoint_every == 0
            ):

                checkpoint_path = (
                    self.checkpoint_dir
                    / f"epoch_{epoch:04d}.pt"
                )

                save_checkpoint(
                    self.generator,
                    self.discriminator,
                    self.g_optimizer,
                    self.d_optimizer,
                    epoch,
                    checkpoint_path,
                )

                print(
                    f"Checkpoint saved: "
                    f"{checkpoint_path}"
                )