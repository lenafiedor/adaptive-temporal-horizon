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
    from adaptive_horizon.data.santafe import SANTAFE

    if system == SANTAFE:
        return observed_fsle_tolerance(
            config.SANTAFE_DATA_PATH, fraction, fallback, n=n
        )
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
    return _tolerance_from_result(result, fraction, fallback, n, t_max, path)


def _tolerance_from_result(result, fraction, fallback, n, t_max, path):
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


# --- Measured data -----------------------------------------------------------
# Without equations to integrate, a perturbed run is replaced by an analogue: a
# nearby state elsewhere in the training recording (outside a Theiler window)
# that is followed forward along the data (Boffetta et al. 2002, Phys. Rep.).


def observed_separation_curves(
    states, n=2000, neighbours=5, t_max=400, theiler_window=50, seed=0
):
    """Window RMSE between analogue pairs, (t_max + 1, pairs), plus FLOPs.

    ``states`` are normalized delay windows of shape (N, history). Each of
    ``n`` reference states is paired with its ``neighbours`` nearest distinct
    states whose continuation of ``t_max`` steps lies inside the recording.
    """
    states = np.asarray(states, dtype=np.float64)
    count, history = states.shape
    usable = count - t_max
    if usable <= 2 * theiler_window + neighbours:
        raise ValueError("Recording is too short for an analogue FSLE")
    rng = np.random.default_rng(seed)
    refs = np.sort(rng.choice(usable, size=min(n, usable), replace=False))
    candidates = states[:usable]
    pairs = []
    for start in range(0, len(refs), 256):
        chunk = refs[start : start + 256]
        dist = np.sqrt(
            ((states[chunk, None, :] - candidates[None, :, :]) ** 2).mean(axis=2)
        )
        dist[np.abs(chunk[:, None] - np.arange(usable)) <= theiler_window] = np.inf
        dist[dist == 0] = np.inf
        nearest = np.argpartition(dist, neighbours - 1, axis=1)[:, :neighbours]
        for row, ref in enumerate(chunk):
            for j in nearest[row]:
                if np.isfinite(dist[row, j]):
                    pairs.append((ref, j))
    pairs = np.asarray(pairs)
    steps = np.arange(t_max + 1)
    a = states[pairs[:, 0, None] + steps]  # (pairs, t_max + 1, history)
    b = states[pairs[:, 1, None] + steps]
    curves = np.sqrt(((a - b) ** 2).mean(axis=2)).T
    flops = (
        len(refs) * usable * 3 * history  # neighbour search
        + len(pairs) * (t_max + 1) * 3 * history  # separation curves
    )
    return curves, int(flops)


def observed_fsle(curves, thresholds, min_pairs=50):
    """lambda(delta_n) per step from analogue pairs.

    Only pairs that start below delta_n time the doubling from delta_n to
    delta_{n+1}; pairs never reaching both thresholds are left out, and a
    threshold needs ``min_pairs`` valid pairs with over half of them crossing.
    """
    initial = curves[0]
    times = []
    for delta in thresholds:
        crossed = curves > delta
        times.append(np.where(crossed.any(0), crossed.argmax(0), np.nan))
    times = np.asarray(times, dtype=float)
    r = thresholds[1] / thresholds[0]
    lam, valid_pairs = [], []
    for k in range(len(thresholds) - 1):
        valid = initial < thresholds[k]
        dt_k = times[k + 1, valid] - times[k, valid]
        ok = np.isfinite(dt_k)
        valid_pairs.append(int(valid.sum()))
        enough = valid.sum() >= min_pairs and ok.mean() > 0.5 and dt_k[ok].mean() > 0
        lam.append(np.log(r) / dt_k[ok].mean() if enough else np.nan)
    return np.asarray(lam), valid_pairs


def compute_observed_fsle(
    data_path,
    n=2000,
    neighbours=5,
    t_max=400,
    theiler_window=50,
    fractions=(0.75, 0.5, 0.25),
    seed=0,
):
    """Analogue FSLE of a recording's training split (Santa Fe laser).

    Separations are window RMSEs in the training normalization, the units of
    the error-truncation tolerance; time is in recorded samples (dt = 1).
    Measurement noise and data density bound the smallest resolvable delta, so
    the small-delta plateau starts at the typical nearest-analogue distance.
    """
    from adaptive_horizon.data.santafe import load_santafe

    start = perf_counter()
    data = load_santafe(data_path, t_max)
    stats = data["metadata"]["normalization_stats"]
    train = data["trajectories"]["train"].numpy().astype(np.float64)
    states = (train - stats["mean"][0]) / (stats["std"][0] + 1e-8)
    curves, flops = observed_separation_curves(
        states, n, neighbours, t_max, theiler_window, seed
    )

    delta_min = float(np.quantile(curves[0], 0.25))
    thresholds = delta_min * np.sqrt(2) ** np.arange(40)
    thresholds = thresholds[thresholds < 1.5]
    lam, valid_pairs = observed_fsle(curves, thresholds)
    sizes = thresholds[:-1] * np.sqrt(2) ** 0.5
    finite = np.where(np.isfinite(lam))[0]
    if len(finite) == 0:
        raise ValueError("No threshold has enough analogue pairs for an FSLE")
    plateau = float(np.nanmedian(lam[finite[:3]]))
    crossover = {}
    for frac in fractions:
        below = np.where(lam < frac * plateau)[0]
        crossover[str(frac)] = float(sizes[below[0]]) if len(below) else None
    return {
        "system": "santafe-laser",
        "data_checksum": data["metadata"]["checksum"],
        "dt": 1.0,
        "n": n,
        "neighbours": neighbours,
        "theiler_window": theiler_window,
        "t_max": t_max,
        "seed": seed,
        "pairs": int(curves.shape[1]),
        "lambda_max_plateau": plateau,
        "lyapunov_time_steps": 1 / plateau,
        "saturation_rmse": float(np.median(curves[-10:])),
        "delta_star": crossover,
        "delta": sizes.tolist(),
        "fsle": lam.tolist(),
        "fsle_over_plateau": (lam / plateau).tolist(),
        "valid_pairs": valid_pairs,
        "thresholds": thresholds.tolist(),
        "flops": flops,
        "wall_time_seconds": perf_counter() - start,
    }


def observed_fsle_tolerance(
    data_path,
    fraction=config.FSLE_KNEE_FRACTION,
    fallback=config.FSLE_FALLBACK_TOLERANCE,
    n=2000,
    t_max=400,
):
    """``fsle_tolerance`` for a recording, cached per file checksum."""
    from adaptive_horizon.data.santafe import SANTAFE, load_santafe

    checksum = load_santafe(data_path, t_max)["metadata"]["checksum"]
    data_dir = config.system_path(config.DATA_DIR, SANTAFE)
    path = data_dir / f"fsle_{SANTAFE}_{checksum[:12]}_n{n}_t{t_max:g}.json"
    if path.exists():
        result = json.loads(path.read_text())
    else:
        result = compute_observed_fsle(
            data_path, n=n, t_max=t_max, fractions=(fraction,)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(path)
    return _tolerance_from_result(result, fraction, fallback, n, t_max, path)
