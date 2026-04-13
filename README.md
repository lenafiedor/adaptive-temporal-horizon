# Adaptive Temporal Horizon in Auto-Regressive Models

This repository contains an extension of the ideas presented in [Adaptive Temporal Horizon in Auto-Regressive Models](https://arxiv.org/abs/2506.03889).

Specifically, we implement an adaptive temporal horizon for training a multi-layer perceptron (MLP) to learn Lorenz attractor dynamics.

## Requirements

- Python 3.13
- Poetry package manager

## Usage

### Setup

```bash
cd adaptive-temporal-horizon
python -m venv ./.venv
source ./.venv/bin/activate
pip install poetry
poetry install
```

### MLP Training and Evaluation

> [!NOTE]
> MLP architecture is strongly inspired by [Temporal horizons in forecasting](https://github.com/vboussange/temporal_horizons_in_forecasting) repository.

First, train the MLP to learn Lorenz attractor dynamics.

```bash
poetry run train-mlp              # With static training horizon (T=1)
poetry run train-mlp --adaptive   # With adaptive training horizon
```

**Args:**

| Name                  | Description                                          | Default value |
|-----------------------|------------------------------------------------------|---------------|
| `--adaptive`, `-a`    | Whether to use the adaptive training horizon or not. | False         |
| `-T`                  | Training horizon (only in fixed horizon mode).       | 1             |
| `--epochs`, `-e`      | Number of epochs to train the model.                 | 100           |

Then, let's evaluate equation (3) from the [Temporal horizons in forecasting: an accuracy-learnability trade-off](https://arxiv.org/abs/2506.03889) paper on the learned model.

```bash
poetry run evaluate-mlp --model=<path/to/trained/model.pt>
```

### Computing Lyapunov Exponents

To analyze Lyapunov exponents, run the following command:

```bash
poetry run compute-lyapunov [--mode=global|local] [--plot]
```

**Args:**
- `--mode`: Specifies whether to compute global or local Lyapunov exponents.
- `--plot`: If set, generates a plot of the Lorenz attractor dynamics. In `local` mode, Lyapunov exponents plot will be generated regardless of this argument.
