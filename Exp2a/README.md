# Autoencoders & Variational Autoencoders Experiment

Implements the six variants shown in the experiment sheet:

1. Shallow Autoencoder
2. Deep Autoencoder
3. Sparse Autoencoder
4. Vanilla VAE
5. Beta-VAE
6. Conditional VAE

## Dataset

MNIST is downloaded automatically by `dataset.py`.

## Project tree

```text
ae_vae_experiment/
├── config.yaml
├── dataset.py
├── models.py
├── utils.py
├── train.py
├── evaluate.py
├── run_all.py
├── requirements.txt
├── data/
└── outputs/
    ├── shallow_ae/
    ├── deep_ae/
    ├── sparse_ae/
    ├── vae/
    ├── beta_vae/
    └── cvae/
```

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or:

```bash
pip install -r requirements.txt
```

## Train one model

```bash
python train.py --model deep_ae
```

## Train every model

```bash
python run_all.py
```

## Evaluate

```bash
python evaluate.py \
  --model deep_ae \
  --checkpoint outputs/deep_ae/best.pt
```

## Metrics

Autoencoders do not have a normal classification accuracy. This project reports:

- Reconstruction BCE/loss
- MSE
- Binary pixel reconstruction accuracy

Pixel accuracy is the percentage of pixels whose thresholded reconstruction agrees with the thresholded original image. For reporting model quality, reconstruction loss and MSE should be considered together with pixel accuracy.

## Expected behavior

On MNIST, the deep AE should generally reconstruct digits very well. VAEs trade some sharpness for a structured probabilistic latent space. Beta-VAE usually gives more disentangled latent representations but can have slightly worse reconstruction. Sparse AE encourages a mostly inactive latent representation. CVAE allows reconstruction/generation conditioned on the digit label.
