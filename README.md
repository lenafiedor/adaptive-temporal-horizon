# Adaptive Temporal Horizon in Auto-Regressive Models

This repository contains an extension of the ideas presented in [Adaptive Temporal Horizon in Auto-Regressive Models](https://arxiv.org/abs/2506.03889).

Specifically, we implement an adaptive temporal horizon for training a multi-layer perceptron (MLP) to learn Lorenz attractor dynamics.

## Requirements

- Python 3.13
- Poetry package manager

## Before you begin

### Setup

```bash
cd adaptive-temporal-horizon
python -m venv ./.venv
source ./.venv/bin/activate
pip install poetry
poetry install
```

From now on, we will use the `poetry` command wrapper to run scripts.

### Linter & Formatter

We use [Ruff package](https://docs.astral.sh/ruff/) for linting and formatting.

Before committing, please make sure to run the following commands:

```bash
poetry run ruff check
poetry run ruff format
```

## Usage

### MLP Training

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


### MLP Evaluation

#### Model-specific gradient scaling

Let's evaluate equation (3) from the [Temporal horizons in forecasting: an accuracy-learnability trade-off](https://arxiv.org/abs/2506.03889) paper on the learned model.

This script will evaluate the model on a range of evaluation horizons by computing the following function:

$$
g(T) = \frac{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, T) \right\|}{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, 1) \right\|}
$$

Which represents the gradient scaling with respect to $T$.

```bash
poetry run evaluate-mlp --model=path/to/trained/model.pt
```

#### Cross-validation on all trained models

This script will evaluate all trained models (`experiments/lorenz/models`) on a set of evaluation horizons.

```bash
poetry run evaluate-mlp --mode=cross-val
```

**Args:**

| Name            | Description                                           | Default value |
|-----------------|-------------------------------------------------------|---------------|
| `--max-eval-T`  | Maximum evaluation horizon to consider for evaluation | 20            |

### Computing Lyapunov Exponents

To analyze Lyapunov exponents, run the following command:

```bash
poetry run compute-lyapunov [--mode=global|local] [--plot]
```

**Args:**

| Name     | Description                                                                            | Default value |
|----------|----------------------------------------------------------------------------------------|---------------|
| `--mode` | Specifies whether to compute global or local Lyapunov exponents. (`global` or `local`) | global        |
| `--plot` | If set, generates a plot of the Lorenz attractor dynamics.                             | False         |
