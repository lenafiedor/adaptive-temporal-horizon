from adaptive_horizon.visualization.plot_budget_resources import EvalScope
from adaptive_horizon.visualization.plot_optimal_horizon import (
    infer_epochs,
    select_seed_train_T,
)


def test_infer_epochs_from_model_directory():
    assert infer_epochs("models/dt_08_T10_05_epochs/fixed", 20) == 5


def test_infer_epochs_for_budget_based_directory():
    assert infer_epochs("models/budget_based_dt_08_fixed/fixed", 20) == 20


def test_select_seed_train_T_overall():
    records = [
        {"model_type": "fixed", "seed": 2, "train_T": 1, "val_T": 1, "mse": 0.4},
        {"model_type": "fixed", "seed": 2, "train_T": 1, "val_T": 2, "mse": 0.6},
        {"model_type": "fixed", "seed": 2, "train_T": 2, "val_T": 1, "mse": 0.2},
        {"model_type": "fixed", "seed": 2, "train_T": 2, "val_T": 2, "mse": 0.3},
    ]

    assert select_seed_train_T(records, 2, "median", EvalScope("overall")) == 2


def test_select_seed_train_T_for_single_validation_horizon():
    records = [
        {"model_type": "fixed", "seed": 2, "train_T": 1, "val_T": 1, "mse": 0.1},
        {"model_type": "fixed", "seed": 2, "train_T": 1, "val_T": 2, "mse": 0.6},
        {"model_type": "fixed", "seed": 2, "train_T": 2, "val_T": 1, "mse": 0.2},
        {"model_type": "fixed", "seed": 2, "train_T": 2, "val_T": 2, "mse": 0.3},
    ]

    assert select_seed_train_T(records, 2, "median", EvalScope("single", 1)) == 1
