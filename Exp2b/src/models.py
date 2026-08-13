import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASELINE GAN
# ============================================================


class GANGenerator(nn.Module):
    """
    Baseline GAN Generator

    Input:
        z: (batch_size, latent_dim)

    Output:
        image: (batch_size, 1, 28, 28)
    """

    def __init__(self, config):
        super().__init__()

        latent_dim = config["training"]["latent_dim"]
        image_size = config["dataset"]["image_size"]
        channels = config["dataset"]["channels"]

        hidden_units = (
            config["models"]["gan"]["generator"]["hidden_units"]
        )

        layers = []

        input_dim = latent_dim

        for units in hidden_units:
            layers.extend([
                nn.Linear(input_dim, units),
                nn.BatchNorm1d(units),
                nn.LeakyReLU(0.2, inplace=True),
            ])

            input_dim = units

        # Final layer
        layers.extend([
            nn.Linear(
                input_dim,
                image_size * image_size * channels
            ),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

        self.image_size = image_size
        self.channels = channels

    def forward(self, z):

        x = self.model(z)

        x = x.view(
            x.size(0),
            self.channels,
            self.image_size,
            self.image_size,
        )

        return x


class GANDiscriminator(nn.Module):
    """
    Baseline GAN Discriminator

    Input:
        image: (batch_size, 1, 28, 28)

    Output:
        logits: (batch_size, 1)
    """

    def __init__(self, config):
        super().__init__()

        image_size = config["dataset"]["image_size"]
        channels = config["dataset"]["channels"]

        hidden_units = (
            config["models"]["gan"]["discriminator"]["hidden_units"]
        )

        input_dim = (
            image_size
            * image_size
            * channels
        )

        layers = [
            nn.Flatten()
        ]

        for units in hidden_units:
            layers.extend([
                nn.Linear(input_dim, units),
                nn.LeakyReLU(0.2, inplace=True),
            ])

            input_dim = units

        # No Sigmoid here.
        # BCEWithLogitsLoss handles it internally.
        layers.append(
            nn.Linear(input_dim, 1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, image):
        return self.model(image)


# ============================================================
# CONDITIONAL GAN (cGAN)
# ============================================================


class CGANGenerator(nn.Module):
    """
    Conditional GAN Generator

    Inputs:
        z      : (batch_size, latent_dim)
        labels : (batch_size,)

    Output:
        image  : (batch_size, 1, 28, 28)
    """

    def __init__(self, config):
        super().__init__()

        self.latent_dim = config["training"]["latent_dim"]

        self.num_classes = (
            config["dataset"]["num_classes"]
        )

        self.image_size = (
            config["dataset"]["image_size"]
        )

        self.channels = (
            config["dataset"]["channels"]
        )

        hidden_units = (
            config["models"]["cgan"]["generator"]["hidden_units"]
        )

        layers = []

        input_dim = (
            self.latent_dim
            + self.num_classes
        )

        for units in hidden_units:
            layers.extend([
                nn.Linear(input_dim, units),
                nn.BatchNorm1d(units),
                nn.LeakyReLU(0.2, inplace=True),
            ])

            input_dim = units

        layers.extend([
            nn.Linear(
                input_dim,
                self.image_size
                * self.image_size
                * self.channels,
            ),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, z, labels):

        # Convert class labels:
        #
        # 0 -> [1,0,0,...]
        # 1 -> [0,1,0,...]
        #
        label_one_hot = F.one_hot(
            labels,
            num_classes=self.num_classes,
        ).float()

        # Concatenate:
        #
        # Noise + Class Label
        #
        x = torch.cat(
            [z, label_one_hot],
            dim=1,
        )

        x = self.model(x)

        x = x.view(
            x.size(0),
            self.channels,
            self.image_size,
            self.image_size,
        )

        return x


class CGANDiscriminator(nn.Module):
    """
    Conditional GAN Discriminator

    Inputs:
        image  : (batch_size, 1, 28, 28)
        labels : (batch_size,)

    Output:
        logits : (batch_size, 1)
    """

    def __init__(self, config):
        super().__init__()

        self.num_classes = (
            config["dataset"]["num_classes"]
        )

        image_size = (
            config["dataset"]["image_size"]
        )

        channels = (
            config["dataset"]["channels"]
        )

        hidden_units = (
            config["models"]["cgan"]["discriminator"]["hidden_units"]
        )

        image_dim = (
            image_size
            * image_size
            * channels
        )

        input_dim = (
            image_dim
            + self.num_classes
        )

        layers = []

        for units in hidden_units:
            layers.extend([
                nn.Linear(input_dim, units),
                nn.LeakyReLU(0.2, inplace=True),
            ])

            input_dim = units

        # No Sigmoid.
        layers.append(
            nn.Linear(input_dim, 1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, image, labels):

        # Flatten image
        image = image.view(
            image.size(0),
            -1,
        )

        # One-hot class label
        label_one_hot = F.one_hot(
            labels,
            num_classes=self.num_classes,
        ).float()

        # Image + Label
        x = torch.cat(
            [image, label_one_hot],
            dim=1,
        )

        return self.model(x)


# ============================================================
# DCGAN
# ============================================================


class DCGANGenerator(nn.Module):
    """
    Deep Convolutional GAN Generator

    Architecture:

        Noise 100
           |
        Dense
           |
       7x7x256
           |
       ConvTranspose
           |
      14x14x128
           |
       ConvTranspose
           |
       28x28x64
           |
         Conv2D
           |
       28x28x1
    """

    def __init__(self, config):
        super().__init__()

        latent_dim = (
            config["training"]["latent_dim"]
        )

        base_filters = (
            config["models"]["dcgan"]
            ["generator"]["base_filters"]
        )

        filter_1 = (
            config["models"]["dcgan"]
            ["generator"]["filter_1"]
        )

        filter_2 = (
            config["models"]["dcgan"]
            ["generator"]["filter_2"]
        )

        self.model = nn.Sequential(

            # ------------------------------------------------
            # Noise -> 7x7x256
            # ------------------------------------------------

            nn.Linear(
                latent_dim,
                7 * 7 * base_filters,
            ),

            nn.BatchNorm1d(
                7 * 7 * base_filters
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Unflatten(
                1,
                (
                    base_filters,
                    7,
                    7,
                ),
            ),

            # ------------------------------------------------
            # 7x7 -> 14x14
            # ------------------------------------------------

            nn.ConvTranspose2d(
                base_filters,
                filter_1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                filter_1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            # ------------------------------------------------
            # 14x14 -> 28x28
            # ------------------------------------------------

            nn.ConvTranspose2d(
                filter_1,
                filter_2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                filter_2
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            # ------------------------------------------------
            # 28x28x64 -> 28x28x1
            # ------------------------------------------------

            nn.Conv2d(
                filter_2,
                1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),

            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator

    Architecture:

        28x28x1
           |
        Conv2D
           |
        14x14x64
           |
        Conv2D
           |
        7x7x128
           |
        Flatten
           |
        Real/Fake
    """

    def __init__(self, config):
        super().__init__()

        filter_1 = (
            config["models"]["dcgan"]
            ["discriminator"]["filter_1"]
        )

        filter_2 = (
            config["models"]["dcgan"]
            ["discriminator"]["filter_2"]
        )

        dropout = (
            config["models"]["dcgan"]
            ["discriminator"]["dropout"]
        )

        self.model = nn.Sequential(

            # ------------------------------------------------
            # 28x28 -> 14x14
            # ------------------------------------------------

            nn.Conv2d(
                1,
                filter_1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Dropout(
                dropout
            ),

            # ------------------------------------------------
            # 14x14 -> 7x7
            # ------------------------------------------------

            nn.Conv2d(
                filter_1,
                filter_2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Dropout(
                dropout
            ),

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            nn.Flatten(),

            nn.Linear(
                filter_2 * 7 * 7,
                1,
            ),
        )

    def forward(self, image):
        return self.model(image)


# ============================================================
# MODEL FACTORY
# ============================================================
def initialize_weights(model):
    """
    DCGAN-style weight initialization.
    """

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Linear,
                nn.Conv2d,
                nn.ConvTranspose2d,
            ),
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0.0,
                )

        elif isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
            ),
        ):

            nn.init.normal_(
                module.weight,
                mean=1.0,
                std=0.02,
            )

            nn.init.constant_(
                module.bias,
                0.0,
            )
            
def build_models(model_name, config):

    model_name = model_name.lower()

    if model_name == "gan":

        generator = GANGenerator(config)
        discriminator = GANDiscriminator(config)

    elif model_name == "cgan":

        generator = CGANGenerator(config)
        discriminator = CGANDiscriminator(config)

    elif model_name == "dcgan":

        generator = DCGANGenerator(config)
        discriminator = DCGANDiscriminator(config)

    else:

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Expected: gan, cgan or dcgan."
        )

    # Initialize weights
    initialize_weights(generator)
    initialize_weights(discriminator)

    return generator, discriminator