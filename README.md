# Adaptive Temporal Horizon in Auto-Regressive Models

This repository contains an extension of the ideas presented in [Adaptive Temporal Horizon in Auto-Regressive Models](https://arxiv.org/abs/2506.03889).

Specifically, we implement an adaptive temporal horizon for training a multi-layer perceptron (MLP) to learn chaotic dynamical-system trajectories such as Lorenz and Rossler.

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

Train MLPs to learn the selected dynamical system. By default, the command trains both fixed-horizon models for `T=1..max_T` and adaptive models.

```bash
poetry run train-mlp                            # Train MLPs with both fixed and adaptive training horizon
poetry run train-mlp --single                   # Train a single model with T = 1
poetry run train-mlp --single -T 10             # Train a single model with T = 10
poetry run train-mlp --single --adaptive        # Train a single adaptive model
poetry run train-mlp --fixed                    # Train only with fixed T
poetry run train-mlp --fixed --max-T 8          # Train fixed models for T = 1..8
poetry run train-mlp --adaptive                 # Train only with adaptive T using adaptive-horizon method
poetry run train-mlp --budget-based --max-T 10  # Train fixed/adaptive models under the same epoch budget
poetry run train-mlp --budget-based --adaptive --adaptive-method adaptive-horizon --max-T 6 --fixed-dir experiments/lorenz/models/budget_based_dt_08_fixed
poetry run train-mlp --system rossler           # Train on Rossler dynamics
poetry run train-mlp --output-dir runs/demo     # Save this run under a custom model_root
```

**Args:**

| Name                | Description                                                                       | Values                                                        | Default value        |
|---------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------|----------------------|
| `--epochs` `-e`     | Number of training epochs                                                         | int                                                           | `config.EPOCHS`      |
| `--single`          | Train a single model; combine with `--adaptive` for adaptive training             | true \| false                                                 | false                |
| `-T`                | Training horizon for fixed `--single` mode                                        | int                                                           | 1                    |
| `--fixed`, `-f`     | Train only fixed-horizon models                                                   | true \| false                                                 | false                |
| `--adaptive`, `-a`  | Train only adaptive models                                                        | true \| false                                                 | false                |
| `--adaptive-method` | Adaptive training method                                                          | `adaptive-horizon` \| `weighted-loss` \| `curriculum-horizon` | see notes            |
| `--fixed-dir`       | Fixed model directory used for budget wall-clock metadata                         | path                                                          | None                 |
| `--max-T`           | Maximum horizon used in aggregate training                                        | int                                                           | `config.MAX_TRAIN_T` |
| `--budget-based`    | Train fixed and adaptive models under one budget                                  | true \| false                                                 | false                |
| `--epochs-per-T`    | Budget mode epochs for each fixed horizon                                         | int                                                           | 20                   |
| `--n-seeds` `-s`    | Number of seeds for aggregate training                                            | int                                                           | `config.NUM_SEEDS`   |
| `--dt`              | Time step for the system simulation                                               | float                                                         | `config.DT`          |
| `--system`          | Dynamical system to train on                                                      | `lorenz` \| `rossler`                                         | `config.SYSTEM`      |
| `--batch-size`      | Batch size for training and validation loaders                                    | int                                                           | `config.BATCH_SIZE`  |
| `--early-stopping`  | Enable early stopping based on online cross-validation                            | true \| false                                                 | false                |
| `--output-dir`      | Directory to save models to; existing directories are reused                      | path                                                          | None                 |
| `--debug`           | Save extra loss and gradient diagnostics                                          | true \| false                                                 | false                |

Notes:
- `--fixed` and `--adaptive` are mutually exclusive. With neither flag, both fixed and adaptive models are trained.
- `--adaptive-method` defaults to `adaptive-horizon` outside budget mode and `curriculum-horizon` in budget mode.
- `--max-T` controls aggregate fixed horizons and the maximum horizon available to adaptive methods.
- `-T` only affects fixed-horizon `--single` training.
- When `--output-dir` points to an existing run, training checks seeds `0..n_seeds-1` and only trains missing models.
- In `--budget-based` mode, fixed models train for `epochs_per_T` epochs and adaptive models train for `epochs_per_T * max_T` epochs.
- With `--budget-based --adaptive-method adaptive-horizon`, adaptive training also stops when it reaches the summed mean fixed-model wall time for `T=1..max_T`, read from `--fixed-dir` or the current run's `fixed/` directory.
- `--early-stopping` applies only to curriculum-horizon adaptive training. At each horizon boundary it evaluates validation horizons `1..max_T`, caches the current model when the median improves, and restores the previous cached model when the median worsens.
- To permanently change default variables, edit `config.toml`.

### Gradient Scaling

Evaluate the gradient scaling ratio from the temporal-horizon paper:

$$
g(T) = \frac{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, T) \right\|}{\left\| \nabla_{\theta} \mathcal{L}_x(\theta, 1) \right\|}
$$

```bash
poetry run gradient-scaling --model path/to/trained/model.pt
poetry run gradient-scaling --model path/to/trained/model.pt --max-eval-T 100 --dt 0.04
poetry run gradient-scaling --model path/to/trained/model.pt --system rossler --per-batch
```

**Args:**

| Name            | Description                               | Values                | Default value       |
|-----------------|-------------------------------------------|-----------------------|---------------------|
| `--model`, `-m` | Path to the trained model                 | str                   | required            |
| `--max-eval-T`  | Maximum evaluation horizon                | int                   | `config.MAX_EVAL_T` |
| `--dt`          | Time step for the system simulation       | float                 | `config.DT`         |
| `--system`      | Dynamical system to evaluate              | `lorenz` \| `rossler` | `config.SYSTEM`     |
| `--per-batch`   | Compute per-batch gradient scaling ratios | true \| false         | false               |

Plots use median plus 95% CI for repeated values.

### Cross-Validation

Evaluate fixed and adaptive models across validation horizons. The command saves a JSON file containing metadata, a median/95% CI summary, and raw `evaluation_records`.

```bash
poetry run cross-validation
poetry run cross-validation --model-dir experiments/lorenz/models/dt_08_20260607_120000
poetry run cross-validation --model-dir experiments/lorenz/models/budget_based_dt_08_T10_20260610_120000
poetry run cross-validation --adaptive-method curriculum-horizon --max-train-T 6
poetry run cross-validation --cached experiments/lorenz/evaluation/mse_results_dt_08_20260607_120000.json
poetry run cross-validation --system rossler
```

**Args:**

| Name                | Description                                                            | Values                                                        | Default value                   |
|---------------------|------------------------------------------------------------------------|---------------------------------------------------------------|---------------------------------|
| `--model-dir`       | Run directory containing `fixed/` and `adaptive/` model subdirectories | str                                                           | Read from `models/last_run.txt` |
| `--fixed-dir`       | Directory with fixed models                                            | str                                                           | Read from `--model-dir`         |
| `--max-train-T`     | Maximum fixed training horizon to include                              | int                                                           | Max fixed T found               |
| `--max-eval-T`      | Maximum validation horizon to evaluate                                 | int                                                           | `config.MAX_EVAL_T`             |
| `--adaptive-method` | Evaluate only adaptive models trained with this method                 | `adaptive-horizon` \| `weighted-loss` \| `curriculum-horizon` | None                            |
| `--cached`          | Reuse a saved cross-validation JSON                                    | str                                                           | None                            |
| `--metric`          | Statistic shown in plots                                               | `mean` \| `median`                                            | `median`                        |
| `--system`          | Dynamical system to evaluate                                           | `lorenz` \| `rossler`                                         | `config.SYSTEM`                 |

Notes:
- Cross-validation infers `dt` from the model directory name.
- `--cached` requires a JSON path.
- Cross-validation expects model directory to contain `fixed/` and `adaptive/` subdirectories (unless `--fixed-dir` is specified).
- Cross-validation always writes the JSON report, MSE plot, MSE seed-subplot plot, and paired-delta plot when fixed and adaptive records are present.

### Budget Training

Budget training is now part of `train-mlp`; cross-validation stays in `cross-validation`.

```bash
poetry run train-mlp --budget-based --max-T 8 --epochs-per-T 20 --n-seeds 10
poetry run cross-validation --adaptive-method curriculum-horizon
```

Notes:
- Budget runs are saved under `models/budget_based_dt_*_T*/fixed` and `models/budget_based_dt_*_T*/adaptive`.
- `models/last_run.txt` points at the budget run root, so `cross-validation` can be run without `--model-dir` immediately after budget training.

### Budget Comparison

Plot adaptive MSE and the best fixed-model MSE as a function of the budget run's `max_train_T`.
The helper reads generated `budget_mse_results_*.json` files and writes both a PNG plot and a CSV with the plotted values.

```bash
poetry run budget-comparison --results-dir experiments/lorenz/evaluation/budget_based
poetry run budget-comparison --results-dir experiments/lorenz/evaluation/budget_based --metric mean --eval-scope T1
```

**Args:**

| Name            | Description                                      | Values                        | Default value     |
|-----------------|--------------------------------------------------|-------------------------------|-------------------|
| `--results-dir` | Directory containing `budget_mse_results_*.json` | str                           | `config.EVAL_DIR` |
| `--metric`      | Statistic shown in plots                         | `mean` \| `median`            | `median`          |
| `--scope`       | Evaluation scope used for plotted MSE values     | `overall` \| `single <T_val>` | `overall`         |
| `--output`      | Output path                                      | str                           | Auto-generated    |

Notes:
- The default `overall` scope plots aggregate MSE across all validation horizons.
- `--eval-scope T1` keeps the same best fixed model selected by the overall comparison, but plots its MSE at validation horizon `T=1` (`0.08` when `dt=0.08`).

### Lyapunov Exponents

Calculate and plot the chosen system with Lyapunov exponents heatmap.

```bash
poetry run compute-lyapunov
poetry run compute-lyapunov --system rossler --mode local --plot
```

**Args:**

| Name           | Description                         | Values                | Default value   |
|----------------|-------------------------------------|-----------------------|-----------------|
| `--mode`, `-m` | Lyapunov computation mode           | `global` \| `local`   | `global`        |
| `--plot`, `-p` | Plot the system trajectory          | true \| false         | false           |
| `--system`     | Dynamical system to analyze         | `lorenz` \| `rossler` | `config.SYSTEM` |
| `--dt`         | Time step for the system simulation | float                 | `config.DT`     |
| `--steps`      | Trajectory length                   | int                   | 10000           |

### Gradient Heatmap

Compute and plot local gradient-scaling values along a diagnostic trajectory.

```bash
poetry run gradient-heatmap --model path/to/trained/model.pt
poetry run gradient-heatmap --model path/to/trained/model.pt --system rossler --T-val 8 --microbatch-size 4 --regenerate
```

**Args:**

| Name                | Description                                    | Values                | Default value             |
|---------------------|------------------------------------------------|-----------------------|---------------------------|
| `--model`, `-m`     | Path to the trained model                      | str                   | required                  |
| `--T-val`           | Evaluation horizon                             | int                   | `config.MAX_EVAL_T`       |
| `--dt`              | Time step for the diagnostic simulation        | float                 | `config.DT`               |
| `--system`          | Dynamical system for the diagnostic trajectory | `lorenz` \| `rossler` | `config.SYSTEM`           |
| `--steps`           | Post-burn-in diagnostic trajectory length      | int                   | `config.TRAJECTORY_STEPS` |
| `--seed`            | Diagnostic trajectory seed                     | int                   | `config.RANDOM_SEED`      |
| `--microbatch-size` | Samples per local gradient-scaling estimate    | int                   | 1                         |
| `--regenerate`      | Regenerate the cached diagnostic trajectory    | true \| false         | false                     |
