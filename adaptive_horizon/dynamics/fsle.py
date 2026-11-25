"""Finite-size Lyapunov exponent (FSLE) and the tolerance it implies.

For an ensemble of attractor states, a tiny random perturbation is integrated
with the true dynamics and its size is measured as the RMSE in the training
normalization (the units of the error-truncation tolerance). For thresholds
delta_n = delta_min r^n,

    lambda(delta_n) = ln r / < t(delta_{n+1}) - t(delta_n) >

where t(delta) is the first time the separation exceeds delta (Aurell et al.
1997). lambda(delta) equals the maximal Lyapunov exponent for small delta and
falls to zero as the separation saturates. The knee delta*_f, where
lambda(delta) first drops below a fraction f of the small-delta plateau, marks
the end of exponential error growth; error truncation uses it as tolerance.
"""

import json
import warnings
from time import perf_counter

import numpy as np

from adaptive_horizon import config
from adaptive_horizon.config import resolve_burn_in_steps
from adaptive_horizon.dynamics.systems import get_system

RHS_FLOPS = {"lorenz": 8, "rossler": 7}
RK4_COMBINE_FLOPS_PER_DIM = 14
RMSE_FLOPS_PER_DIM = 4


def batched_rhs(system):
    """Vector field on states of shape (dim, n)."""
    system = get_system(system)
    if system.name.startswith("lorenz96"):
        F = system.parameters["forcing"]
        return lambda x: (
            (np.roll(x, -1, 0) - np.roll(x, 2, 0)) * np.roll(x, 1, 0) - x + F
        )
    return system.rhs


def rk4(f, x, h):
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def separation_curves(system, x0, std, delta0, t_max, h, rng):
    """Normalized RMSE between reference and perturbed runs, (n_times, n)."""
    f = batched_rhs(system)
    n, d = x0.shape
    u = rng.standard_normal((n, d))
    u *= delta0 / np.sqrt(np.mean(u**2, axis=1, keepdims=True))
    a, b = x0.T.copy(), (x0 + u * std).T.copy()
    out = []
    for _ in range(round(t_max / h)):
        a, b = rk4(f, a, h), rk4(f, b, h)
        out.append(np.sqrt(np.mean(((a - b).T / std) ** 2, axis=1)))
    return np.asarray(out)


def fsle(curves, h, thresholds):
    """lambda(delta_n) from first-crossing times; NaN where too few cross."""
    times = []
    for delta in thresholds:
        crossed = curves > delta
        t = np.where(crossed.any(0), crossed.argmax(0) * h + h, np.nan)
        times.append(t)
    times = np.asarray(times)
    r = thresholds[1] / thresholds[0]
    lam = []
    for k in range(len(thresholds) - 1):
        dt_k = times[k + 1] - times[k]
        ok = np.isfinite(dt_k)
        lam.append(np.log(r) / dt_k[ok].mean() if ok.mean() > 0.5 else np.nan)
    with warnings.catch_warnings():  # thresholds no trajectory reaches
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_times = np.nanmean(times, axis=1)
    return np.asarray(lam), mean_times


def fsle_flops(system, n, t_max, h=config.INTEGRATION_DT):
    """Floating-point operations of separation_curves (two RK4 runs + RMSE)."""
    system = get_system(system)
    dim = system.dim
    rhs = 4 * dim if system.name.startswith("lorenz96") else RHS_FLOPS[system.name]
    per_step = 2 * (4 * rhs + RK4_COMBINE_FLOPS_PER_DIM * dim)
    per_step += RMSE_FLOPS_PER_DIM * dim
    return int(n * round(t_max / h) * per_step)


def compute_fsle(
    system,
    dt,
    n=2000,
    delta0=1e-8,
    t_max=40.0,
    fractions=(0.75, 0.5, 0.25),
    seed=0,
):
    """FSLE curve and knees in the training normalization of (system, dt)."""
    from adaptive_horizon.data.dataset import TrajectoryDataset
    from adaptive_horizon.data.utils import default_split_gap

    start = perf_counter()
    rng = np.random.default_rng(seed)
    common = {
        "T": 1,
        "dt": dt,
        "system": system,
        "split_gap": default_split_gap(1),
        "burn_in": resolve_burn_in_steps(dt),
    }
    stats = TrajectoryDataset(split="train", **common).normalization_stats
    std = np.asarray(stats["std"], dtype=float) + 1e-8
    ds = TrajectoryDataset(split="val", normalization_stats=stats, **common)
    raw = ds.raw_trajectory.numpy().astype(float)
    x0 = raw[np.linspace(0, len(raw) - 1, n).astype(int)]

    h = config.INTEGRATION_DT
    curves = separation_curves(system, x0, std, delta0, t_max, h, rng)
    thresholds = delta0 * 100 * np.sqrt(2) ** np.arange(60)
    thresholds = thresholds[thresholds < 1.5]
    lam, t_cross = fsle(curves, h, thresholds)
    sizes = thresholds[:-1] * np.sqrt(2) ** 0.5
    plateau = float(np.nanmedian(lam[:5]))
    crossover = {}
    for frac in fractions:
        below = np.where(lam < frac * plateau)[0]
        crossover[str(frac)] = float(sizes[below[0]]) if len(below) else None
    saturation = float(np.median(curves[-round(2 / h) :]))
    steps = (t_cross - t_cross[0]) / dt
    return {
        "system": get_system(system).name,
        "dt": dt,
        "n": n,
        "delta0": delta0,
        "t_max": t_max,
        "seed": seed,
        "lambda_max_plateau": plateau,
        "lyapunov_time_steps": 1 / (plateau * dt),
        "saturation_rmse": saturation,
        "delta_star": crossover,
        "delta": sizes.tolist(),
        "fsle": lam.tolist(),
        "fsle_over_plateau": (lam / plateau).tolist(),
        "thresholds": thresholds.tolist(),
        "mean_first_crossing_steps_from_first": steps.tolist(),
        "flops": fsle_flops(system, n, t_max, h),
        "wall_time_seconds": perf_counter() - start,
    }


def fsle_tolerance(
    system,
    dt,
    fraction=config.FSLE_KNEE_FRACTION,
    fallback=config.FSLE_FALLBACK_TOLERANCE,
    n=2000,
    t_max=40.0,
):
    """Truncation tolerance from the FSLE knee, with its cost.

    tau = delta*_fraction; if lambda(delta) never drops below that fraction
    before saturation (no knee), tau = ``fallback``. The FSLE is deterministic
    (fixed seed), so the result is cached next to the trajectory data; the cost
    (FLOPs) is reported either way, since every model that uses tau pays it.
    """
    system = get_system(system)
    data_dir = config.system_path(config.DATA_DIR, system.name)
    path = data_dir / f"fsle_{system.name}_dt{dt:g}_n{n}_t{t_max:g}.json"
    if path.exists():
        result = json.loads(path.read_text())
    else:
        result = compute_fsle(system.name, dt, n=n, t_max=t_max, fractions=(fraction,))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(path)
    knee = result["delta_star"].get(str(fraction))
    if knee is None and str(fraction) not in result["delta_star"]:
        lam = np.asarray(result["fsle_over_plateau"], dtype=float)
        below = np.where(lam < fraction)[0]
        knee = float(result["delta"][below[0]]) if len(below) else None
    return {
        "tolerance": float(knee if knee is not None else fallback),
        "source": "fsle_knee" if knee is not None else "fsle_fallback",
        "knee_fraction": fraction,
        "knee": knee,
        "fallback": fallback,
        "lambda_max": result["lambda_max_plateau"],
        "lyapunov_time_steps": result["lyapunov_time_steps"],
        "flops": result["flops"],
        "wall_time_seconds": result["wall_time_seconds"],
        "n": n,
        "t_max": t_max,
        "cache": str(path),
    }
