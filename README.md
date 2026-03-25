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
python src/train.py
```
