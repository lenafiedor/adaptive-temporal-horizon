from pathlib import Path
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
DATA_DIR = Path(_config["paths"]["data_dir"])

DEVICE = _config["training"]["device"]
MAX_TRAIN_T = _config["training"]["max_train_T"]
EPOCHS = _config["training"]["epochs"]
BATCH_SIZE = _config["training"]["batch_size"]
OPTIMIZER = _config["training"]["optimizer"]
LEARNING_RATE = _config["training"]["learning_rate"]
WEIGHT_DECAY = _config["training"]["weight_decay"]
NUM_SEEDS = _config["training"]["num_seeds"]
TRAJECTORY_STEPS = _config["training"]["trajectory_steps"]
TRAJECTORY_SEED = _config["training"]["trajectory_seed"]
TRAIN_FRACTION = _config["training"]["train_fraction"]
DT = _config["training"]["dt"]
BURN_IN_TIME = _config["training"]["burn_in_time"]
VARIANCE = _config["training"]["variance"]
DEFAULT_ADAPTIVE_HORIZON = _config["training"]["default_adaptive_horizon"]

RHO = _config["weighted-loss"]["rho"]
TEMPERATURE = _config["weighted-loss"]["temperature"]
WEIGHT_FLOOR = _config["weighted-loss"]["weight_floor"]
ANCHOR_ALPHA = _config["weighted-loss"]["anchor_alpha"]
FTLE_WINDOW = _config["weighted-loss"]["ftle_window"]

CURRICULUM_LOSS_THRESHOLD = _config["curriculum-horizon"]["curriculum_loss_threshold"]
CURRICULUM_UPDATE_FREQUENCY = _config["curriculum-horizon"][
    "curriculum_update_frequency"
]
CURRICULUM_EARLY_STOPPING = _config["curriculum-horizon"]["early_stopping"]
CURRICULUM_EARLY_STOP_PATIENCE = _config["curriculum-horizon"]["early_stop_patience"]
CURRICULUM_EARLY_STOP_MIN_DELTA = _config["curriculum-horizon"]["early_stop_min_delta"]
CURRICULUM_EARLY_STOP_MIN_T = _config["curriculum-horizon"]["early_stop_min_T"]
CURRICULUM_EARLY_STOP_GRACE_EPOCHS = _config["curriculum-horizon"][
    "early_stop_grace_epochs"
]

INPUT_DIM = _config["model"]["input_dim"]
LAYER_WIDTH = _config["model"]["layer_width"]

NUM_BATCHES = _config["evaluation"]["num_batches"]
EVAL_SEED = _config["evaluation"]["eval_seed"]
MAX_EVAL_T = _config["evaluation"]["max_eval_T"]
