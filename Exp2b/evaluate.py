import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from scipy.linalg import sqrtm

from src.models import (
    GANGenerator,
    CGANGenerator,
    DCGANGenerator,
)


# ============================================================
# FASHION-MNIST CLASSIFIER
# ============================================================


class FashionClassifier(nn.Module):
    """
    CNN classifier used only for evaluation.

    It provides:
        1. Class predictions
        2. Class probabilities
        3. Feature embeddings for FID
    """

    def __init__(self, feature_dim=128):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.feature_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 10),
        )

    def extract_features(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)

        return self.classifier[0](x)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)

        features = self.classifier[0](x)
        features = self.classifier[1](features)

        logits = self.classifier[2](features)

        return logits, features


# ============================================================
# CLASSIFIER TRAINING
# ============================================================


def train_classifier(
    device,
    data_dir="./data",
    epochs=5,
    batch_size=128,
    lr=1e-3,
    save_path="outputs/evaluation/fashion_classifier.pt",
):
    """
    Train a Fashion-MNIST classifier if one does not exist.
    """

    print("\nTraining Fashion-MNIST evaluation classifier...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,),
        ),
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = FashionClassifier().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _ = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        test_acc = evaluate_classifier_accuracy(
            model,
            test_loader,
            device,
        )

        print(
            f"Classifier Epoch "
            f"{epoch:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        save_path,
    )

    print(
        f"\nEvaluation classifier saved: "
        f"{save_path}"
    )

    return model


def evaluate_classifier_accuracy(
    model,
    loader,
    device,
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


# ============================================================
# LOAD CLASSIFIER
# ============================================================


def load_classifier(
    device,
    checkpoint_path,
    data_dir,
):

    model = FashionClassifier().to(device)

    if os.path.exists(checkpoint_path):

        print(
            f"Loading evaluation classifier: "
            f"{checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint
        )

    else:

        model = train_classifier(
            device=device,
            data_dir=data_dir,
            save_path=checkpoint_path,
        )

    model.eval()

    return model


# ============================================================
# LOAD GENERATOR
# ============================================================


def load_generator(
    model_name,
    config,
    checkpoint_path,
    device,
):

    model_name = model_name.lower()

    if model_name == "gan":

        generator = GANGenerator(config)

    elif model_name == "cgan":

        generator = CGANGenerator(config)

    elif model_name == "dcgan":

        generator = DCGANGenerator(config)

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Support several checkpoint formats
    if "generator_state_dict" in checkpoint:

        state_dict = checkpoint[
            "generator_state_dict"
        ]

    elif "generator" in checkpoint:

        state_dict = checkpoint[
            "generator"
        ]

    else:

        state_dict = checkpoint

    generator.load_state_dict(
        state_dict
    )

    generator.to(device)
    generator.eval()

    return generator


# ============================================================
# GENERATE IMAGES
# ============================================================


@torch.no_grad()
def generate_images(
    generator,
    model_name,
    latent_dim,
    num_samples,
    device,
):

    all_images = []
    all_labels = []

    batch_size = 256

    remaining = num_samples

    while remaining > 0:

        current_batch = min(
            batch_size,
            remaining,
        )

        z = torch.randn(
            current_batch,
            latent_dim,
            device=device,
        )

        if model_name == "cgan":

            labels = torch.randint(
                0,
                10,
                (
                    current_batch,
                ),
                device=device,
            )

            images = generator(
                z,
                labels,
            )

            all_labels.append(
                labels.cpu()
            )

        else:

            images = generator(z)

        all_images.append(
            images.cpu()
        )

        remaining -= current_batch

    images = torch.cat(
        all_images,
        dim=0,
    )

    if all_labels:

        labels = torch.cat(
            all_labels,
            dim=0,
        )

    else:

        labels = None

    return images, labels


# ============================================================
# REAL FASHION-MNIST
# ============================================================


def load_real_images(
    data_dir,
    num_samples,
):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,),
        ),
    ])

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
    )

    images = []
    labels = []

    count = 0

    for batch_images, batch_labels in loader:

        remaining = (
            num_samples - count
        )

        if remaining <= 0:
            break

        batch_images = batch_images[
            :remaining
        ]

        batch_labels = batch_labels[
            :remaining
        ]

        images.append(
            batch_images
        )

        labels.append(
            batch_labels
        )

        count += batch_images.size(0)

    return (
        torch.cat(images),
        torch.cat(labels),
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================


@torch.no_grad()
def extract_features(
    model,
    images,
    device,
    batch_size=256,
):

    features = []

    for i in range(
        0,
        len(images),
        batch_size,
    ):

        batch = images[
            i:i + batch_size
        ].to(device)

        _, batch_features = model(
            batch
        )

        features.append(
            batch_features.cpu()
        )

    return torch.cat(
        features,
        dim=0,
    ).numpy()


# ============================================================
# PREDICTIONS
# ============================================================


@torch.no_grad()
def classify_images(
    model,
    images,
    device,
):

    probabilities = []

    predictions = []

    for i in range(
        0,
        len(images),
        256,
    ):

        batch = images[
            i:i + 256
        ].to(device)

        logits, _ = model(batch)

        probs = F.softmax(
            logits,
            dim=1,
        )

        probabilities.append(
            probs.cpu()
        )

        predictions.append(
            probs.argmax(dim=1).cpu()
        )

    probabilities = torch.cat(
        probabilities
    )

    predictions = torch.cat(
        predictions
    )

    return (
        probabilities,
        predictions,
    )


# ============================================================
# FID
# ============================================================


def calculate_fid(
    real_features,
    fake_features,
):

    real_features = np.asarray(
        real_features,
        dtype=np.float64,
    )

    fake_features = np.asarray(
        fake_features,
        dtype=np.float64,
    )

    mu_real = np.mean(
        real_features,
        axis=0,
    )

    mu_fake = np.mean(
        fake_features,
        axis=0,
    )

    sigma_real = np.cov(
        real_features,
        rowvar=False,
    )

    sigma_fake = np.cov(
        fake_features,
        rowvar=False,
    )

    diff = (
        mu_real - mu_fake
    )

    covmean = sqrtm(
        sigma_real @ sigma_fake
    )

    if np.iscomplexobj(covmean):

        covmean = covmean.real

    fid = (
        diff.dot(diff)
        + np.trace(
            sigma_real
            + sigma_fake
            - 2 * covmean
        )
    )

    return float(fid)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================


def calculate_class_metrics(
    probabilities,
    predictions,
):

    counts = torch.bincount(
        predictions,
        minlength=10,
    ).float()

    distribution = (
        counts / counts.sum()
    )

    # Number of classes represented
    class_coverage = int(
        (counts > 0).sum().item()
    )

    # Entropy
    entropy = -torch.sum(
        distribution
        * torch.log(
            distribution + 1e-8
        )
    )

    normalized_entropy = (
        entropy
        / math.log(10)
    )

    # KL divergence from uniform
    uniform = torch.full(
        (10,),
        0.1,
    )

    kl_divergence = torch.sum(
        distribution
        * torch.log(
            (distribution + 1e-8)
            / uniform
        )
    )

    # Confidence
    confidence = (
        probabilities.max(
            dim=1
        ).values.mean()
    )

    return {
        "class_coverage": class_coverage,
        "class_entropy": float(
            entropy
        ),
        "normalized_class_entropy": float(
            normalized_entropy
        ),
        "kl_divergence_to_uniform": float(
            kl_divergence
        ),
        "classifier_confidence": float(
            confidence
        ),
        "class_distribution": (
            distribution.tolist()
        ),
    }


# ============================================================
# INCEPTION-LIKE SCORE
# ============================================================


def calculate_inception_like_score(
    probabilities,
):

    p_y = probabilities.mean(
        dim=0,
        keepdim=True,
    )

    kl = probabilities * (
        torch.log(
            probabilities + 1e-8
        )
        - torch.log(
            p_y + 1e-8
        )
    )

    score = torch.exp(
        kl.sum(dim=1).mean()
    )

    return float(score)


# ============================================================
# PIXEL DIVERSITY
# ============================================================


def calculate_pixel_diversity(
    images,
    num_pairs=5000,
):

    images = images.view(
        len(images),
        -1,
    )

    n = len(images)

    indices_a = torch.randint(
        0,
        n,
        (num_pairs,),
    )

    indices_b = torch.randint(
        0,
        n,
        (num_pairs,),
    )

    distances = torch.norm(
        images[indices_a]
        - images[indices_b],
        dim=1,
    )

    return float(
        distances.mean()
    )


# ============================================================
# CONDITIONAL ACCURACY
# ============================================================


def calculate_conditional_accuracy(
    predictions,
    requested_labels,
):

    correct = (
        predictions
        == requested_labels
    ).float()

    return float(
        correct.mean()
    )


# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(
    results,
    output_dir,
    model_name,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    json_path = (
        Path(output_dir)
        / f"{model_name}_evaluation.json"
    )

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        f"\nResults saved to: "
        f"{json_path}"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GAN, cGAN and DCGAN "
            "on Fashion-MNIST."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "gan",
            "cgan",
            "dcgan",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--data-dir",
        default="./data",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--classifier",
        default=(
            "outputs/evaluation/"
            "fashion_classifier.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    import yaml

    with open(
        args.config,
        "r",
    ) as f:

        config = yaml.safe_load(f)

    # --------------------------------------------------------
    # Evaluation classifier
    # --------------------------------------------------------

    classifier = load_classifier(
        device=device,
        checkpoint_path=args.classifier,
        data_dir=args.data_dir,
    )

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    print(
        f"\nLoading {args.model.upper()}..."
    )

    generator = load_generator(
        model_name=args.model,
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # --------------------------------------------------------
    # Generate images
    # --------------------------------------------------------

    print(
        f"\nGenerating "
        f"{args.num_samples} images..."
    )

    fake_images, requested_labels = (
        generate_images(
            generator,
            args.model,
            config["training"]["latent_dim"],
            args.num_samples,
            device,
        )
    )

    # --------------------------------------------------------
    # Real images
    # --------------------------------------------------------

    print(
        "Loading real Fashion-MNIST "
        "images..."
    )

    real_images, real_labels = (
        load_real_images(
            args.data_dir,
            args.num_samples,
        )
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    print(
        "Extracting features..."
    )

    real_features = extract_features(
        classifier,
        real_images,
        device,
    )

    fake_features = extract_features(
        classifier,
        fake_images,
        device,
    )

    # --------------------------------------------------------
    # FID
    # --------------------------------------------------------

    print(
        "Calculating FID..."
    )

    fid = calculate_fid(
        real_features,
        fake_features,
    )

    # --------------------------------------------------------
    # Class predictions
    # --------------------------------------------------------

    print(
        "Classifying generated images..."
    )

    fake_probabilities, fake_predictions = (
        classify_images(
            classifier,
            fake_images,
            device,
        )
    )

    # --------------------------------------------------------
    # Class metrics
    # --------------------------------------------------------

    class_metrics = (
        calculate_class_metrics(
            fake_probabilities,
            fake_predictions,
        )
    )

    # --------------------------------------------------------
    # Inception-like score
    # --------------------------------------------------------

    inception_score = (
        calculate_inception_like_score(
            fake_probabilities
        )
    )

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    diversity = (
        calculate_pixel_diversity(
            fake_images
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "model": args.model,

        "num_samples": (
            args.num_samples
        ),

        "fid": fid,

        "inception_like_score": (
            inception_score
        ),

        "pixel_diversity": (
            diversity
        ),

        **class_metrics,
    }

    # --------------------------------------------------------
    # cGAN conditional accuracy
    # --------------------------------------------------------

    if (
        args.model == "cgan"
        and requested_labels is not None
    ):

        conditional_accuracy = (
            calculate_conditional_accuracy(
                fake_predictions,
                requested_labels,
            )
        )

        results[
            "conditional_accuracy"
        ] = conditional_accuracy

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        f"EVALUATION RESULTS: "
        f"{args.model.upper()}"
    )
    print("=" * 70)

    print(
        f"FID                         : "
        f"{fid:.4f}"
    )

    print(
        f"Inception-like Score       : "
        f"{inception_score:.4f}"
    )

    print(
        f"Pixel Diversity             : "
        f"{diversity:.4f}"
    )

    print(
        f"Classifier Confidence      : "
        f"{class_metrics['classifier_confidence']:.4f}"
    )

    print(
        f"Class Coverage             : "
        f"{class_metrics['class_coverage']}/10"
    )

    print(
        f"Class Entropy               : "
        f"{class_metrics['class_entropy']:.4f}"
    )

    print(
        f"Normalized Class Entropy    : "
        f"{class_metrics['normalized_class_entropy']:.4f}"
    )

    print(
        f"KL Divergence to Uniform    : "
        f"{class_metrics['kl_divergence_to_uniform']:.4f}"
    )

    if (
        args.model == "cgan"
        and "conditional_accuracy"
        in results
    ):

        print(
            f"Conditional Accuracy        : "
            f"{results['conditional_accuracy']:.4f}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        args.output_dir,
        args.model,
    )


if __name__ == "__main__":
    main()