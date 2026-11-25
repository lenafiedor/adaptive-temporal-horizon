"""Finite-size Lyapunov exponent (FSLE) and the error-truncation tolerance.

Prints lambda(delta) relative to its small-delta plateau, the knee delta* and
the tolerance that train.py --tolerance-from-fsle would use for the given
system and timestep. See adaptive_horizon.dynamics.fsle for the method.

Usage: python -m adaptive_horizon.analysis.compute_fsle --system lorenz96 --dt 0.1
"""

import argparse
import json
from pathlib import Path

from adaptive_horizon import config
from adaptive_horizon.dynamics.fsle import compute_fsle, fsle_tolerance
from adaptive_horizon.dynamics.systems import SYSTEM_CHOICES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", choices=SYSTEM_CHOICES, default=config.DEFAULT_SYSTEM
    )
    parser.add_argument("--dt", type=float, default=config.DT, help="Model timestep")
    parser.add_argument("--n", type=int, default=2000, help="Ensemble size")
    parser.add_argument("--delta0", type=float, default=1e-8)
    parser.add_argument("--t-max", type=float, default=40.0)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[config.FSLE_KNEE_FRACTION, 0.5, 0.25],
        help="Knee fractions of lambda_max to report",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Save the result as JSON"
    )
    args = parser.parse_args()

    r = compute_fsle(
        args.system,
        args.dt,
        n=args.n,
        delta0=args.delta0,
        t_max=args.t_max,
        fractions=tuple(args.fractions),
    )
    knees = ", ".join(
        f"{k}: {v:.2f}" if v is not None else f"{k}: none"
        for k, v in r["delta_star"].items()
    )
    print(
        f"{args.system} dt={args.dt}: lambda_max={r['lambda_max_plateau']:.3f} "
        f"(Lyapunov time {r['lyapunov_time_steps']:.1f} steps), "
        f"saturation RMSE={r['saturation_rmse']:.2f}, delta*: {knees}"
    )
    for delta, ratio in zip(r["delta"], r["fsle_over_plateau"]):
        if delta > 0.01:
            print(f"  delta={delta:6.3f}  lambda/lambda_max={ratio:5.2f}")
    if args.n == 2000 and args.t_max == 40.0:
        tol = fsle_tolerance(args.system, args.dt)
        print(f"tolerance: {tol['tolerance']:.3f} ({tol['source']})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
