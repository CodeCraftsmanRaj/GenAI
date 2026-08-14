import torch
from torch import nn


class ShallowAE(nn.Module):
    def __init__(self, d=784, z=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, z), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(z, d), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x.flatten(1))
        return self.decoder(z), z


class DeepAE(nn.Module):
    def __init__(self, d=784, z=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, z), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(z, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, d), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.flatten(1))
        return self.decoder(z), z


class SparseAE(DeepAE):
    pass


class VAE(nn.Module):
    def __init__(self, d=784, z=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, z)
        self.logvar = nn.Linear(256, z)
        self.decoder = nn.Sequential(
            nn.Linear(z, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, d), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class BetaVAE(VAE):
    pass


class CVAE(nn.Module):
    def __init__(self, d=784, z=16, classes=10):
        super().__init__()
        self.classes = classes
        self.encoder = nn.Sequential(
            nn.Linear(d + classes, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, z)
        self.logvar = nn.Linear(256, z)
        self.decoder = nn.Sequential(
            nn.Linear(z + classes, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, d), nn.Sigmoid()
        )

    def one_hot(self, y):
        return torch.nn.functional.one_hot(y, self.classes).float()

    def encode(self, x, y):
        h = self.encoder(torch.cat([x.flatten(1), self.one_hot(y)], 1))
        return self.mu(h), self.logvar(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        inp = torch.cat([z, self.one_hot(y)], 1)
        return self.decoder(inp), mu, logvar


def build_model(name, cfg):
    d = cfg["model"]["input_dim"]
    z = cfg["model"]["latent_dim"]
    c = cfg["model"]["num_classes"]
    models = {
        "shallow_ae": ShallowAE,
        "deep_ae": DeepAE,
        "sparse_ae": SparseAE,
        "vae": VAE,
        "beta_vae": BetaVAE,
        "cvae": CVAE,
    }
    return models[name](d, z, c) if name == "cvae" else models[name](d, z)
