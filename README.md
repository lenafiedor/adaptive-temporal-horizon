# Adaptive Temporal Horizon in Auto-Regressive Models

## Requirements

- Python 3.13

## Usage

### Create the virtual environment

```bash
cd adaptive_temporal_horizon
python -m venv ./.venv
source ./.venv/bin/activate
pip intall -r requirements.txt
```

### Train the MLP to learn Lorenz attractor dynamics

```bash
python src/train.py [-T=<temporal_horizon>]
```

### Evaluate Equation (3) from the [Temporal horizons in forecasting: an accuracy-learnability trade-off](https://arxiv.org/abs/2506.03889) paper

```bash
python src/evaluate.py --model=<model_path>
```

### Compute Global or Local Lyapunov Exponents

```bash
python src/scripts/compute_lyapunov_exponents.py [--mode=global|local]
```
