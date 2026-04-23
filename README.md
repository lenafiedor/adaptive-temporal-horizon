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

From now on, we will use the `poetry run` command wrapper to run scripts.

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

Train MLPs to learn Lorenz attractor dynamics.
The script trains fixed-horizon MLPs for `T=1..max_T` and can also train adaptive-horizon models.

```bash
poetry run train-mlp                            # Train MLPs with both fixed and adaptive training horizon
poetry run train-mlp --single                   # Train a single model with T = 1
poetry run train-mlp --single -T 10             # Train a single model with T = 10
poetry run train-mlp --fixed                    # Train only with fixed T
poetry run train-mlp --fixed --max-T 8          # Train fixed models for T = 1..8
poetry run train-mlp --adaptive                 # Train only with adaptive T
poetry run train-mlp --fixed --append --max-T 8 # Append only missing fixed T values to the last run
```

**Args:**

| Name               | Description                                                   | Values        | Default value   |
|--------------------|---------------------------------------------------------------|---------------|-----------------|
| `--epochs` `-e`    | Number of epochs to train the model                           | int           | 100             |
| `--single`         | Train a single fixed-horizon model                            | true \| false | false           |
| `-T`               | Training horizon used with `--single`                         | int           | 1               |
| `--fixed`, `-f`    | Train only the fixed horizon models                           | true \| false | false           |
| `--adaptive`, `-a` | Train only the adaptive horizon models                        | true \| false | false           |
| `--max-T`          | Train fixed-horizon models for all `T` from 1 to this value   | int           | 10              |
| `--n-seeds` `-s`   | Number of seeds to use for aggregate training                 | int           | 10              |
| `--dt`             | Time step for the Lorenz attractor simulation                 | float         | 0.04            |
| `--append`         | Append outputs to the run stored in `last_run.txt`            | true \| false | false           |


The trained models are saved in the `experiments/lorenz/models/<timestamp>` directory by default.
There should be 10 models by default (10 seeds x (7 horizons + adaptive)) after running the aggregate training script.

Notes:
- `--max-T` only affects fixed-horizon training in aggregate mode. It is ignored when using `--single`.
- `-T` only affects fixed horizon training in single mode. It is ignored when using `--adaptive` or `--fixed`.
- `--append` reuses the full model path stored in `experiments/lorenz/models/last_run.txt`.
- In `--append` mode, training checks seeds `0..n_seeds-1` and only trains the missing ones for each fixed `T` and for adaptive models.

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
poetry run gradient-scaling --model=path/to/trained/model.pt
```

**Args:**

| Name           | Description                                            | Values | Default value |
|----------------|--------------------------------------------------------|--------|---------------|
| `--model` `-m` | Path to the trained model                              | str    | Required      |
| `--max-eval-T` | Maximum evaluation horizon to consider for evaluation  | int    | 200           |
| `--dt`         | Time step for the Lorenz attractor simulation          | float  | 0.04          |

#### Cross-validation on all trained models

This script will evaluate all trained models from the last test run (timestamp saved at `experiments/lorenz/models/last_run.txt`) on a set of evaluation horizons.

T values for evaluation are dynamically set to the same as found trained models, but you can also specify a maximum value.
Models will be then additionally validated at each T value divisible by 10 that is greater than the maximum T found in the training set.

**Example:**
- You have trained models with `T = [1, 2, 4, 8, 12, 16, 20]`
- `max_eval_T` is set to 100
- Each model will be evaluated with `T = [1, 2, 4, 8, 12, 16, 20, 30, 40, 50, 60, 70, 80, 90, 100]`

```bash
poetry run cross-validation
```

**Args:**

| Name            | Description                                           | Values | Default value                      |
|-----------------|-------------------------------------------------------|--------|------------------------------------|
| `--model-dir`   | Path to the directory containing trained models       | str    | Read from `last_run.txt`           |
| `--max-train-T` | Maximum training horizon to consider for evaluation   | int    | Max T found in the model directory |
| `--max-eval-T`  | Maximum evaluation horizon to consider for evaluation | int    | Max T found in the model directory |
| `--dt`          | Time step for the Lorenz attractor simulation         | float  | 0.04                               |

### Computing Lyapunov Exponents

To analyze Lyapunov exponents, run the following command:

```bash
poetry run compute-lyapunov [--mode=global|local] [--plot]
```

**Args:**

| Name             | Description                                                     | Values          | Default value |
|------------------|-----------------------------------------------------------------|-----------------|---------------|
| `--mode`         | Specifies whether to compute global or local Lyapunov exponents | global \| local | global        |
| `--plot`, `-p`   | If set, generates a plot of the Lorenz attractor dynamics       | true \| false   | false         |
| `--window`, `-w` | Window size for computing local Lyapunov exponents              | int             | 10            |
