from adaptive_horizon.visualization.plot_optimal_horizon import infer_epochs


def test_infer_epochs_from_model_directory():
    assert infer_epochs("models/dt_08_T10_05_epochs/fixed", 20) == 5


def test_infer_epochs_for_budget_based_directory():
    assert infer_epochs("models/budget_based_dt_08_fixed/fixed", 20) == 20
