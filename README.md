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

### Single MLP

> [!NOTE]
> MLP architecture is strongly inspired by [Temporal horizons in forecasting](https://github.com/vboussange/temporal_horizons_in_forecasting) repository.

First, train a single MLP to learn Lorenz attractor dynamics.

```bash
poetry run train-mlp              # With static training horizon
poetry run train-mlp --adaptive   # With adaptive training horizon
```

**Args:**

| Name               | Description                                         | Values            | Default value |
|--------------------|-----------------------------------------------------|-------------------|---------------|
| `--adaptive`, `-a` | Whether to use the adaptive training horizon or not | `true` \| `false` | false         |
| `-T`               | Training horizon (only in fixed horizon mode)       | int               | 1             |
| `--epochs`, `-e`   | Number of epochs to train the model                 | int               | 100           |

#### Aggregate Training

To run the aggregate training script, run the following command:

```bash
poetry run train-aggregate-mlp                  # All training horizons + adaptive training horizon
poetry run train-aggregate-mlp --adaptive-only  # Only adaptive training horizon
```

The script will iterate over temporal horizons and seeds specified in the `config.toml` file and train the MLP for each combination.
Seeds are fixed for reproducibility.

The trained models are saved in the `experiments/lorenz/models` directory by default.
There should be 10 models by default (10 seeds x (7 horizons + adaptive)) after running the aggregate training script.

To customize the settings, edit `config.toml` directly.

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

> [!NOTE]
> We strongly reccommend to set a larger `max-eval-T` value than the default one (20) to observe the exponential growth of gradient scaling.

#### Cross-validation on all trained models

This script will evaluate all trained models (`experiments/lorenz/models`) on a set of evaluation horizons.

T values for evaluation are dynamically set to the same as found trained models, but you can also specify a maximum value.
Models will be then additionally validated at each T value divisible by 10 that is greater than the maximum T found in the training set.

**Example:**
- You have trained models with `T = [1, 2, 4, 8, 12, 16, 20]`
- `max_val_T` is set to 100
- Each model will be evaluated with `T = [1, 2, 4, 8, 12, 16, 20, 30, 40, 50, 60, 70, 80, 90, 100]`

```bash
poetry run evaluate-mlp --mode=cross-val
```

**Args:**

| Name           | Description                                           | Values | Default value |
|----------------|-------------------------------------------------------|--------|---------------|
| `--max-eval-T` | Maximum evaluation horizon to consider for evaluation | int    | 20            |

### Computing Lyapunov Exponents

To analyze Lyapunov exponents, run the following command:

```bash
poetry run compute-lyapunov [--mode=global|local] [--plot]
```

**Args:**

| Name     | Description                                                     | Values              | Default value |
|----------|-----------------------------------------------------------------|---------------------|---------------|
| `--mode` | Specifies whether to compute global or local Lyapunov exponents | `global` \| `local` | global        |
| `--plot` | If set, generates a plot of the Lorenz attractor dynamics       | `true` \| `false`   | false         |
