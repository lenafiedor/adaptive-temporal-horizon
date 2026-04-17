import torch
import argparse

from adaptive_horizon.config import MODEL_DIR, TRAIN_TS, SEEDS
from adaptive_horizon.training.train import train, create_model_and_loaders
from adaptive_horizon.visualization.plotting import save_model


def train_single_model(seed, epochs, device, T=None, adaptive=False):
    model, train_loader, val_loader, optimizer, config = create_model_and_loaders(
        seed, adaptive, device, T
    )

    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=epochs,
        device=device,
        T=T,
        adaptive=adaptive,
    )

    return save_model(model, config, seed, MODEL_DIR, T=T, adaptive=adaptive)


def train_adaptive_models(seeds, epochs, device):
    adaptive_paths = []

    print(f"\n{'=' * 50}")
    print("Training adaptive models")
    print(f"{'=' * 50}")

    for seed in seeds:
        print(f"\n--- Adaptive Seed {seed} ---")
        model_path = train_single_model(seed, epochs, device, adaptive=True)
        adaptive_paths.append(model_path)

    return adaptive_paths


def train_all_models(train_Ts, seeds, epochs, device):
    model_paths = {T: [] for T in train_Ts}

    for T in train_Ts:
        print(f"\n{'=' * 50}")
        print(f"Training models for T={T}")
        print(f"{'=' * 50}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            model_path = train_single_model(seed, epochs, device, T=T)
            model_paths[T].append(model_path)

    adaptive_paths = train_adaptive_models(seeds, epochs, device)

    return model_paths, adaptive_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--adaptive-only", '-a',
        action="store_true",
        help="Train only adaptive models (skip fixed-T models)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.adaptive_only:
        train_adaptive_models(SEEDS, args.epochs, device)
    else:
        train_all_models(TRAIN_TS, SEEDS, args.epochs, device)

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
