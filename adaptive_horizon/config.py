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
TRAIN_TS = _config["training"]["train_Ts"]
BATCH_SIZE = _config["training"]["batch_size"]
LEARNING_RATE = _config["training"]["learning_rate"]
WEIGHT_DECAY = _config["training"]["weight_decay"]
LAYER_WIDTH = _config["model"]["layer_width"]
