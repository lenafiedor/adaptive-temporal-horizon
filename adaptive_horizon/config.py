from pathlib import Path
import math
import tomllib


def load_config():
    """Load configuration from config.toml"""
    config_path = Path(__file__).parent.parent / "config.toml"

    with open(config_path, "rb") as f:
        return tomllib.load(f)


_config = load_config()

MODEL_DIR = Path(_config["paths"]["models_dir"])
EVAL_DIR = Path(_config["paths"]["eval_dir"])
LOSS_DIR = Path(_config["paths"]["loss_dir"])
ANALYSIS_DIR = Path(_config["paths"]["analysis_dir"])

DEVICE = _config["training"]["device"]
MAX_T = _config["training"]["max_T"]
EPOCHS = _config["training"]["epochs"]
BATCH_SIZE = _config["training"]["batch_size"]
OPTIMIZER = _config["training"]["optimizer"]
LEARNING_RATE = _config["training"]["learning_rate"]
WEIGHT_DECAY = _config["training"]["weight_decay"]
NUM_SEEDS = _config["training"]["num_seeds"]
NUM_TRAJECTORIES = _config["training"]["num_trajectories"]
STEPS_PER_TRAJECTORY = _config["training"]["steps_per_trajectory"]
DT = _config["training"]["dt"]
BURN_IN_TIME = _config["training"]["burn_in_time"]
DEFAULT_ADAPTIVE_HORIZON = _config["training"]["default_adaptive_horizon"]
RHO = _config["training"]["rho"]
TEMPERATURE = _config["training"]["temperature"]
WEIGHT_FLOOR = _config["training"]["weight_floor"]
ANCHOR_ALPHA = _config["training"]["anchor_alpha"]

INPUT_DIM = _config["model"]["input_dim"]
LAYER_WIDTH = _config["model"]["layer_width"]
WINDOW_SIZE = _config["model"]["window_size"]


def resolve_burn_in_steps(dt=DT, burn_in=None, burn_in_time=BURN_IN_TIME):
    """Resolve transient burn-in steps from physical Lorenz time.

    Passing an explicit integer burn_in keeps that exact value, including zero.
    Leaving burn_in as None uses ceil(burn_in_time / dt).
    """
    if burn_in is not None:
        if burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {burn_in}")
        return int(burn_in)

    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if burn_in_time < 0:
        raise ValueError(f"burn_in_time must be non-negative, got {burn_in_time}")

    return int(math.ceil(burn_in_time / dt))
