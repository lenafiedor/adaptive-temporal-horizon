import argparse

import numpy as np
import pytest

from adaptive_horizon.analysis import compute_gradient_heatmap as heatmap
from adaptive_horizon.dynamics.systems import get_system


def test_validate_checkpoint_accepts_matching_system_and_dt():
    checkpoint = {"metadata": {"system": "rossler", "dt": "0.08000000000000001"}}

    heatmap.validate_checkpoint(checkpoint, get_system("rossler"), 0.08)


@pytest.mark.parametrize(
    ("checkpoint", "match"),
    [
        ({"metadata": {}}, "metadata\\['system'\\]"),
        ({"metadata": {"system": "lorenz", "dt": 0.08}}, "trained for 'lorenz'"),
        ({"metadata": {"system": "rossler"}}, "metadata\\['dt'\\]"),
        ({"metadata": {"system": "rossler", "dt": 0.04}}, "dt=0.04"),
    ],
)
def test_validate_checkpoint_rejects_missing_or_mismatched_metadata(
    checkpoint,
    match,
):
    with pytest.raises(ValueError, match=match):
        heatmap.validate_checkpoint(checkpoint, get_system("rossler"), 0.08)


def test_build_diagnostic_samples_aligns_inputs_targets_and_indices():
    trajectory = np.arange(15, dtype=np.float64).reshape(5, 3)

    inputs, targets, sample_indices = heatmap.build_diagnostic_samples(
        trajectory,
        T_val=2,
    )

    assert inputs.shape == (3, 3)
    assert targets.shape == (3, 2, 3)
    np.testing.assert_array_equal(sample_indices, np.array([0, 1, 2]))
    np.testing.assert_allclose(inputs[0].numpy(), trajectory[0])
    np.testing.assert_allclose(targets[0].numpy(), trajectory[1:3])


def test_build_diagnostic_samples_rejects_too_short_trajectory():
    trajectory = np.arange(6, dtype=np.float64).reshape(2, 3)

    with pytest.raises(ValueError, match="No diagnostic samples"):
        heatmap.build_diagnostic_samples(trajectory, T_val=2)


def test_compute_gradient_heatmap_validates_checkpoint_before_trajectory_load(
    monkeypatch,
):
    class DummyModel:
        def to(self, device):
            return self

    monkeypatch.setattr(
        heatmap,
        "load_model",
        lambda path: (DummyModel(), {"metadata": {"system": "lorenz", "dt": 0.08}}),
    )
    monkeypatch.setattr(
        heatmap,
        "get_checkpoint_normalization_stats",
        lambda checkpoint: None,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("trajectory should not be loaded after validation fails")

    monkeypatch.setattr(heatmap, "get_trajectory", fail_if_called)

    args = argparse.Namespace(
        model="model.pt",
        system="rossler",
        dt=0.08,
        steps=10,
        seed=123,
        T_val=2,
        microbatch_size=1,
        regenerate=False,
    )

    with pytest.raises(ValueError, match="trained for 'lorenz'"):
        heatmap.compute_gradient_heatmap(args)
