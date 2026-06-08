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

Train MLPs to learn Lorenz attractor dynamics. By default, the command trains both fixed-horizon models for `T=1..max_T` and adaptive models.

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

| Name                | Description                                                           | Values                                                                                      | Default value        |
|---------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------|----------------------|
| `--epochs` `-e`     | Number of training epochs                                             | int                                                                                         | `config.EPOCHS`      |
| `--single`          | Train a single model; combine with `--adaptive` for adaptive training | true \| false                                                                               | false                |
| `-T`                | Training horizon for fixed `--single` mode                            | int                                                                                         | 1                    |
| `--fixed`, `-f`     | Train only fixed-horizon models                                       | true \| false                                                                               | false                |
| `--adaptive`, `-a`  | Train only adaptive models                                            | true \| false                                                                               | false                |
| `--adaptive-method` | Adaptive training method                                              | `adaptive-horizon` \| `weighted-loss` \| `curriculum-horizon` \| `gradient-scaling-horizon` | `adaptive-horizon`   |
| `--max-T`           | Maximum horizon used in aggregate training                            | int                                                                                         | `config.MAX_TRAIN_T` |
| `--n-seeds` `-s`    | Number of seeds for aggregate training                                | int                                                                                         | `config.NUM_SEEDS`   |
| `--dt`              | Time step for the Lorenz simulation                                   | float                                                                                       | `config.DT`          |
| `--batch-size`      | Batch size for training and validation loaders                        | int                                                                                         | `config.BATCH_SIZE`  |
| `--append`          | Append missing models to the run referenced by `models/last_run.txt`  | true \| false                                                                               | false                |
| `--debug`           | Save extra loss and gradient diagnostics                              | true \| false                                                                               | false                |

Notes:
- `--fixed` and `--adaptive` are mutually exclusive. With neither flag, both fixed and adaptive models are trained.
- `--max-T` controls aggregate fixed horizons and the maximum horizon available to adaptive methods.
- `-T` only affects fixed-horizon `--single` training.
- In `--append` mode, training checks seeds `0..n_seeds-1` and only trains missing models.
- To permanently change default variables, edit `config.toml`.

### Gradient Scaling

Evaluate the gradient scaling ratio from the temporal-horizon paper:

$$
g(T) = \frac{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, T) \right\|}{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, 1) \right\|}
$$

```bash
poetry run gradient-scaling --model path/to/trained/model.pt
poetry run gradient-scaling --model path/to/trained/model.pt --max-eval-T 100 --dt 0.04
poetry run gradient-scaling --model path/to/trained/model.pt --per-batch
```

**Args:**

| Name             | Description                                  | Values          | Default value        |
|------------------|----------------------------------------------|-----------------|----------------------|
| `--model`, `-m`  | Path to the trained model                    | str             | required             |
| `--max-eval-T`   | Maximum evaluation horizon                   | int             | `config.MAX_EVAL_T`  |
| `--dt`           | Time step for the Lorenz simulation          | float           | `config.DT`          |
| `--per-batch`    | Compute per-batch gradient scaling ratios    | true \| false   | false                |

Plots use median plus 95% CI for repeated values.

### Cross-Validation

Evaluate fixed and adaptive models across validation horizons. The command saves a JSON file containing metadata, a median/95% CI summary, and raw `evaluation_records`.

```bash
poetry run cross-validation
poetry run cross-validation --model-dir experiments/lorenz/models/dt_08_20260607_120000
poetry run cross-validation --fixed-dir path/to/fixed/models --model-dir path/to/adaptive/models
poetry run cross-validation --adaptive-method curriculum-horizon --max-T 6
poetry run cross-validation --cached experiments/lorenz/evaluation/mse_results_dt_08_20260607_120000.json
```

**Args:**

| Name                | Description                                                                                                  | Values                                                                                      | Default value                   |
|---------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------|
| `--model-dir`       | Directory containing adaptive models, and fixed models unless `--fixed-dir` is set                           | str                                                                                         | Read from `models/last_run.txt` |
| `--fixed-dir`       | Directory containing fixed-horizon models                                                                    | str                                                                                         | `--model-dir`                   |
| `--max-T`           | Maximum horizon to evaluate                                                                                  | int                                                                                         | Max fixed T found               |
| `--adaptive-method` | Evaluate only adaptive models trained with this method                                                       | `adaptive-horizon` \| `weighted-loss` \| `curriculum-horizon` \| `gradient-scaling-horizon` | None                            |
| `--cached`          | Reuse fixed-model records from a saved cross-validation JSON and evaluate adaptive models from `--model-dir` | str                                                                                         | None                            |

Notes:
- Cross-validation infers `dt` from the model directory name.
- `--cached` requires a JSON path.

### Budget Training

Train fixed models and curriculum-horizon adaptive models under the same epoch budget, then cross-validate them.

```bash
poetry run budget-train
poetry run budget-train --max-train-T 8 --epochs-per-T 20 --n-seeds 10
poetry run budget-train --cached path/to/model/dir
poetry run budget-train --max-eval-T 6
```

**Args:**

| Name             | Description                                                      | Values | Default value       |
|------------------|------------------------------------------------------------------|--------|---------------------|
| `--dt`           | Time step for the Lorenz simulation                              | float  | `config.DT`         |
| `--max-train-T`  | Maximum training horizon                                         | int    | auto                |
| `--max-eval-T`   | Maximum validation horizon                                       | int    | auto                |
| `--epochs-per-T` | Epochs for each fixed horizon                                    | int    | 20                  |
| `--n-seeds`      | Number of seeds                                                  | int    | `config.NUM_SEEDS`  |
| `--batch-size`   | Batch size for training and validation loaders                   | int    | `config.BATCH_SIZE` |
| `--save-dir`     | Directory for evaluation JSON and plots                          | str    | `config.EVAL_DIR`   |
| `--cached`       | Skip training and cross-validate existing budget model directory | str    | None                |

Notes:
- Without `--cached`, fixed models train for `epochs_per_T` epochs and adaptive models train for `epochs_per_T * max_train_T` epochs.
- With `--cached`, `dt`, available train horizons, and default eval horizon are inferred from the cached model directory.

### Lyapunov Exponents

```bash
poetry run compute-lyapunov
poetry run compute-lyapunov --mode local --plot
```

**Args:**

| Name           | Description                         | Values              | Default value |
|----------------|-------------------------------------|---------------------|---------------|
| `--mode`, `-m` | Lyapunov computation mode           | `global` \| `local` | `global`      |
| `--plot`, `-p` | Plot the Lorenz trajectory          | true \| false       | false         |
| `--dt`         | Time step for the Lorenz simulation | float               | 0.01          |
| `--steps`      | Trajectory length                   | int                 | 10000         |

### Gradient Heatmap

Compute and plot local gradient-scaling values along a diagnostic Lorenz trajectory.

```bash
poetry run gradient-heatmap --model path/to/trained/model.pt
poetry run gradient-heatmap --model path/to/trained/model.pt --T-val 8 --microbatch-size 4 --regenerate
```

**Args:**

| Name                | Description                                 | Values        | Default value                 |
|---------------------|---------------------------------------------|---------------|-------------------------------|
| `--model`, `-m`     | Path to the trained model                   | str           | required                      |
| `--T-val`           | Evaluation horizon                          | int           | `config.MAX_EVAL_T`           |
| `--dt`              | Time step for the Lorenz simulation         | float         | `config.DT`                   |
| `--steps`           | Post-burn-in diagnostic trajectory length   | int           | `config.STEPS_PER_TRAJECTORY` |
| `--seed`            | Diagnostic trajectory seed                  | int           | `config.EVAL_SEED`            |
| `--microbatch-size` | Samples per local gradient-scaling estimate | int           | 1                             |
| `--regenerate`      | Regenerate the cached diagnostic trajectory | true \| false | false                         |
