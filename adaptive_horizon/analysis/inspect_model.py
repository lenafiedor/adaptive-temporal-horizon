import argparse
import json
from pathlib import Path
import torch


def tensor_shapes(state_dict):
    return {name: list(tensor.shape) for name, tensor in state_dict.items()}


def summarize_checkpoint(checkpoint):
    model_config = checkpoint.get("config", {})
    metadata = checkpoint.get("metadata", {})
    state_dict = checkpoint.get("model_state_dict", {})

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
