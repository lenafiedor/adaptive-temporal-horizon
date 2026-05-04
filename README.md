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
poetry run train-mlp --single --adaptive        # Train a single adaptive model
poetry run train-mlp --fixed                    # Train only with fixed T
poetry run train-mlp --fixed --max-T 8          # Train fixed models for T = 1..8
poetry run train-mlp --adaptive                 # Train only with adaptive T using adaptive-horizon method
poetry run train-mlp --fixed --append --max-T 8 # Append only missing fixed T values to the last run
```

**Args:**

| Name                | Description                                                           | Values                                | Default value      |
|---------------------|-----------------------------------------------------------------------|---------------------------------------|--------------------|
| `--epochs` `-e`     | Number of epochs to train the model                                   | int                                   | 100                |
| `--single`          | Train a single model; combine with `--adaptive` for adaptive training | true \| false                         | false              |
| `-T`                | Training horizon for fixed `--single` mode                            | int                                   | 1                  |
| `--fixed`, `-f`     | Train only fixed-horizon models                                       | true \| false                         | false              |
| `--adaptive`, `-a`  | Train only adaptive models                                            | true \| false                         | false              |
| `--adaptive-method` | Adaptive training method used with `--adaptive`                       | `adaptive-horizon` \| `weighted-loss` | `adaptive-horizon` |
| `--max-T`           | Train fixed-horizon models for all `T` from 1 to this value           | int                                   | 10                 |
| `--n-seeds` `-s`    | Number of seeds for aggregate training                                | int                                   | 10                 |
| `--dt`              | Time step for the Lorenz attractor simulation                         | float                                 | 0.08               |
| `--batch-size`      | Batch size for training and validation loaders                        | int                                   | 64                 |
| `--optimizer`       | Optimizer to use                                                      | `sgd` \| `adam` \| `adamw`            | adam               |
| `--append`          | Append outputs to the run stored in `last_run.txt`                    | true \| false                         | false              |
| `--adaptive-T-max`  | Shared rollout horizon for LLE-weighted adaptive training             | int                                   | None               |
| `--ftle-window`     | Forward FTLE window for adaptive lambda scores                        | int                                   | 5                  |
| `--rho`             | Predictability budget threshold for adaptive weights                  | float                                 | 1.0                |
| `--temperature`     | Sigmoid temperature for adaptive weights                              | float                                 | 0.25               |
| `--weight-floor`    | Minimum unnormalized adaptive rollout weight                          | float                                 | 0.05               |
| `--anchor-alpha`    | One-step anchor fraction in the adaptive loss                         | float                                 | 0.25               |


The trained models are saved in the `experiments/lorenz/models/dt_<dt>_<timestamp>` directory by default.
There should be 10 models by default (10 seeds x (7 horizons + adaptive)) after running the aggregate training script.

Notes:
- `--max-T` only affects fixed-horizon training in aggregate mode. It is ignored when using `--single`.
- `-T` only affects fixed-horizon `--single` mode.
- `--single --adaptive` trains one adaptive model with seed `0`.
- `--adaptive-method` is only relevant for adaptive training.
- `--adaptive-T-max`, `--ftle-window`, `--rho`, `--temperature`, `--weight-floor`, and `--anchor-alpha` are only used by the `weighted-loss` adaptive method.
- `--append` reuses the full model path stored in `experiments/lorenz/models/last_run.txt`.
- In `--append` mode, training checks seeds `0..n_seeds-1` and only trains the missing ones for each fixed `T` and for adaptive models.

To permanently change the default variables, edit `config.toml` directly.

### MLP Evaluation

#### Model-specific gradient scaling

Let's evaluate equation (3) from the [Temporal horizons in forecasting: an accuracy-learnability trade-off](https://arxiv.org/abs/2506.03889) paper on the learned model.

This script will evaluate the model on a range of evaluation horizons by computing the following function:

$$
g(T) = \frac{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, T) \right\|}{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, 1) \right\|}
$$

Which represents the gradient scaling with respect to $T$.

```bash
poetry run gradient-scaling --model path/to/trained/model.pt
poetry run gradient-scaling --model path/to/trained/model.pt --max-eval-T 100 --dt 0.04
```

**Args:**

| Name                | Description                                           | Values                                | Default value   |
|---------------------|-------------------------------------------------------|---------------------------------------|-----------------|
| `--model` `-m`      | Path to the trained model                             | str                                   | None (required) |
| `--max-eval-T`      | Maximum evaluation horizon to consider for evaluation | int                                   | 200             |
| `--dt`              | Time step for the Lorenz attractor simulation         | float                                 | 0.08            |

#### Cross-validation on all trained models

This script will evaluate all trained models from the last test run (timestamp saved at `experiments/lorenz/models/last_run.txt`) on a set of evaluation horizons.

T values for evaluation are dynamically set to the same as found trained models, but you can also specify a maximum value of the training horizon to consider for evaluation.

```bash
poetry run cross-validation
poetry run cross-validation --adaptive-method weighted-loss --plot median-iqr
poetry run cross-validation --cached
poetry run cross-validation --cached path/to/mse/results/file.json --plot median-iqr
```

**Args:**

| Name                | Description                                                     | Values                                  | Default value                      |
|---------------------|-----------------------------------------------------------------|-----------------------------------------|------------------------------------|
| `--model-dir`       | Path to the directory containing trained models                 | str                                     | Read from `last_run.txt`           |
| `--max-T`           | Maximum training horizon to consider for evaluation             | int                                     | Max T found in the model directory |
| `--adaptive-method` | Evaluate only adaptive models trained with this method          | `adaptive-horizon` \| `weighted-loss`   | None (both used)                   |
| `--cached`          | Reuse cached cross-validation JSON and only regenerate the plot | `None` \| `filename.json`               | None                               |
| `--plot`            | Plot summary statistic and interval style for the figure        | `mean-std` \| `mean-ci` \| `median-iqr` | `mean-ci`                          |

Notes:
- `cross-validation` infers `dt` from the model directory name rather than taking it as a CLI argument.
- Running `cross-validation` saves raw per-model `evaluation_records` to JSON, so plots can be regenerated later without re-evaluating models.
- `poetry run cross-validation --cached` uses the most recently saved cross-validation JSON from `experiments/lorenz/evaluation/last_run.txt`.
- `poetry run cross-validation --cached filename.json` loads that JSON either from the provided path.
- In cached mode, the command only regenerates the plot; it does not rerun evaluation, and `--model-dir`, `--max-T`, and `--adaptive-method` are ignored.

### Computing Lyapunov Exponents

To analyze Lyapunov exponents, run the following command:

```bash
poetry run compute-lyapunov
poetry run compute-lyapunov --mode local --plot --window 20
```

**Args:**

| Name             | Description                                                     | Values          | Default value |
|------------------|-----------------------------------------------------------------|-----------------|---------------|
| `--mode`         | Specifies whether to compute global or local Lyapunov exponents | global \| local | global        |
| `--plot`, `-p`   | If set, generates a plot of the Lorenz attractor dynamics       | true \| false   | false         |
| `--window`, `-w` | Window size for computing local Lyapunov exponents              | int             | 5             |
| `--dt`           | Time step for the Lorenz attractor simulation                   | float           | 0.08          |
| `--steps`        | Number of steps for the Lorenz trajectory                       | int             | 1000          |
