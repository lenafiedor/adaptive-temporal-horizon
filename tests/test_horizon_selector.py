import torch

from adaptive_horizon.training.schedules import HorizonScheduler
from adaptive_horizon.training.train import HorizonSelector


def _select(losses_by_T, tolerance):
    model = torch.nn.Linear(1, 1)
    selector = HorizonSelector(tolerance=tolerance)
    for epoch, (T, loss) in enumerate(losses_by_T, start=1):
        selector.add(model, loss, T, epoch)
    return selector.select()


def test_prefers_largest_T_within_tolerance():
    selected = _select([(1, 1.00), (3, 1.03), (5, 1.20)], tolerance=0.05)
    assert selected["T"] == 3


def test_zero_tolerance_picks_lowest_loss():
    selected = _select([(1, 1.00), (3, 1.03), (5, 1.20)], tolerance=0.0)
    assert selected["T"] == 1


def test_ties_at_same_T_pick_lowest_loss():
    selected = _select([(1, 1.0), (10, 1.02), (10, 1.01)], tolerance=0.05)
    assert (selected["T"], selected["epoch"]) == (10, 3)


def test_empty_selector_restores_nothing():
    assert not HorizonSelector().restore(torch.nn.Linear(1, 1))


def _stall(**overrides):
    params = {
        "T_max": 3,
        "max_epochs": 6,
        "patience": 2,
        "min_delta": 0.01,
        "min_epochs": 3,
        "smoothing_epochs": 2,
    }
    return HorizonScheduler(**{**params, **overrides})


def test_fixed_schedule_splits_epochs_evenly_and_steps_by_one():
    scheduler = HorizonScheduler.fixed(T_max=10, epochs=200)
    Ts = []
    for _ in range(200):
        Ts.append(scheduler.T)
        scheduler.step()
    assert Ts == [T for T in range(1, 11) for _ in range(20)]
    assert scheduler.stage_score is None


def test_fixed_schedule_with_fewer_epochs_than_T_max():
    scheduler = HorizonScheduler.fixed(T_max=10, epochs=5)
    for _ in range(5):
        assert scheduler.step()
    assert scheduler.T == 6


def test_stall_advances_T_and_records_smoothed_score():
    scheduler = _stall()
    ended = [scheduler.step(loss) for loss in (1.0, 0.5, 0.5, 0.5, 0.5)]
    assert ended == [False] * 4 + [True]
    assert scheduler.stage_score == 0.5
    assert scheduler.T == 2


def test_min_epochs_delays_stall():
    scheduler = _stall(min_epochs=4)
    ended = [scheduler.step(1.0) for _ in range(4)]
    assert ended == [False] * 3 + [True]


def test_max_epochs_forces_boundary_while_improving():
    scheduler = _stall()
    ended = [scheduler.step(1.0 / (i + 1)) for i in range(6)]
    assert ended == [False] * 5 + [True]
    assert scheduler.T == 2


def test_force_boundary_and_T_capped_at_max():
    scheduler = _stall(T_max=1)
    assert scheduler.step(1.0, force_boundary=True)
    assert scheduler.T == 1


def test_smoothing_window_carries_over_when_T_stays_at_max():
    scheduler = _stall(T_max=1)
    scheduler.step(1.0, force_boundary=True)
    scheduler.step(3.0, force_boundary=True)
    assert scheduler.stage_score == 2.0


def test_smoothing_window_resets_when_T_advances():
    scheduler = _stall(T_max=2)
    scheduler.step(1.0, force_boundary=True)
    scheduler.step(3.0, force_boundary=True)
    assert scheduler.stage_score == 3.0
