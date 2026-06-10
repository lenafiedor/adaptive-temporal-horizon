import argparse
import json
from pathlib import Path

import torch

import adaptive_horizon.config as config


def infer_history_window(checkpoint):
    metadata = checkpoint.get("metadata", {})
    if "history_window" in metadata:
        return int(metadata["history_window"]), "metadata.history_window"

    model_config = checkpoint.get("config", {})
    input_size = model_config.get("input_size")
    if input_size is not None:
        return int(
            input_size
        ) // config.INPUT_DIM, "config.input_size / config.INPUT_DIM"

    return int(config.HISTORY_WINDOW), "config.HISTORY_WINDOW fallback"


def tensor_shapes(state_dict):
    return {name: list(tensor.shape) for name, tensor in state_dict.items()}


def summarize_checkpoint(checkpoint):
    model_config = checkpoint.get("config", {})
    metadata = checkpoint.get("metadata", {})
    state_dict = checkpoint.get("model_state_dict", {})
    history_window, history_window_source = infer_history_window(checkpoint)

    input_size = model_config.get("input_size")
    output_size = model_config.get("output_size")
    layer_widths = model_config.get("layer_widths", [])

    return {
        "checkpoint_keys": sorted(checkpoint.keys()),
        "train_T": checkpoint.get("train_T"),
        "seed": checkpoint.get("seed"),
        "model": {
            "input_size": input_size,
            "output_size": output_size,
            "input_dim": config.INPUT_DIM,
            "history_window": history_window,
            "history_window_source": history_window_source,
            "layer_widths": layer_widths,
            "num_hidden_blocks": len(layer_widths),
            "residual_connections": model_config.get("residual_connections"),
            "k": model_config.get("k"),
        },
        "metadata": metadata,
        "state_dict_shapes": tensor_shapes(state_dict),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the saved structure and metadata of an adaptive_horizon .pt model."
    )
    parser.add_argument("model", type=Path, help="Path to a saved .pt checkpoint")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Print checkpoint/config/metadata summary without state_dict tensor shapes",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    summary = summarize_checkpoint(checkpoint)
    if args.metadata_only:
        summary.pop("state_dict_shapes")

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
