# Adaptive Temporal Horizon in Auto-Regressive Models

## Requirements

- Python 3.13

## Usage

### Setup

```bash
cd adaptive_temporal_horizon
python -m venv ./.venv
source ./.venv/bin/activate
pip intall -r requirements.txt
```

### MLP Training and Evaluation

First, train the MLP to learn Lorenz attractor dynamics.

```bash
poetry run train-mlp              # With static training_results horizon (T=1)
poetry run train-mlp --adaptive   # With adaptive training_results horizon
```

Args:

| Name | Description | Default value | Scope (static/adaptive) |
|------|-------------|---------------|-------------------------|
| `-T` | Training horizon or initial T if using an adaptive horizon. | 1 | Both |
| `--epochs`, `-e` | Number of epochs to train the model. | 100 | Both |
| `--adaptive`, `-a` | Whether to use the adaptive training horizon or not. | False | – |
| `--max-T` | Maximum training horizon | 16 | Adaptive |
| `-warmup`, `-w` | Number of warmup epochs. During the warmup period, the training horizon is static and equal to 1. | 10 | Adaptive |
|`--update-freq`, `-u` | Frequency of the adaptive training horizon update. | 5 | Adaptive |

Then, let's evaluate equation (3) from the [Temporal horizons in forecasting: an accuracy-learnability trade-off](https://arxiv.org/abs/2506.03889) paper on the learned model.

```bash
poetry run evaluate-mlp --model=<path/to/trained/model.pt>
```

### Compute Lyapunov Exponents

To analyse Lyapunov exponents, run the following command:

```bash
poetry run compute-lyapunov [--mode=global|local] [--plot]
```

`--mode` argument specifies whether to compute global or local Lyapunov exponents.
