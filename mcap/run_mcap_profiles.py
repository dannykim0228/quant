"""
Run MCAP on a single profile-likelihood CSV using mcap.py.

Usage:
  python run_mcap_profiles.py --file path/to/profile.csv --span 0.75
Ex) 
    python run_mcap_profiles.py --file genPompProfile.csv --span 0.75
    python run_mcap_profiles.py --file panelPompProfile.csv --span 0.75
    python run_mcap_profiles.py --file spacetimeProfile.csv --span 1.0

Optional:
  --level 0.95        Confidence level (default 0.95)
  --n-grid 1000       Grid points for smoothing (default 1000)
  --out results.csv   Write a one-row CSV summary
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mcap import mcap, MCAPResult


def _read_profile_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    param_col = None
    for c in ["parameter", "phi", "par", "theta"]:
        if c in lower_map:
            param_col = lower_map[c]
            break
    if param_col is None:
        raise ValueError(
            f"{path}: could not find a parameter column among {list(df.columns)}"
        )

    ll_col = None
    for c in ["mc_profileloglik", "loglik", "loglikelihood", "profile_loglik", "loglik^p"]:
        if c in lower_map:
            ll_col = lower_map[c]
            break
    if ll_col is None:
        raise ValueError(
            f"{path}: could not find a log-likelihood column among {list(df.columns)}"
        )

    parameter = df[param_col].to_numpy(dtype=float)
    loglik = df[ll_col].to_numpy(dtype=float)
    return parameter, loglik


def _fmt(x: Optional[float], digits: int = 6) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NA"
    return f"{x:.{digits}f}"


def run_one(path: Path, span: float, level: float = 0.95, n_grid: int = 1000) -> Dict[str, float]:
    parameter, loglik = _read_profile_csv(path)
    res: MCAPResult = mcap(
        parameter=parameter,
        loglik=loglik,
        level=level,
        span=span,
        n_grid=n_grid,
        loess_degree=2,
    )

    print(f"\n=== {path.name} ===")
    print(f"points: {len(parameter)}  span: {span}  level: {level}  grid: {n_grid}")
    print(f"param range: [{parameter.min():.6f}, {parameter.max():.6f}]")
    print(f"loglik range: [{np.nanmin(loglik):.6f}, {np.nanmax(loglik):.6f}]")
    print(f"smoothed MLE (argmax on grid): {_fmt(res.mle)}")
    print(f"quadratic max:                  {_fmt(res.quadratic_max)}")
    ci_lo, ci_hi = res.ci
    print(f"95% CI (MCAP):                 [{_fmt(ci_lo)}, {_fmt(ci_hi)}]")
    print(f"delta (cutoff):                 {_fmt(res.delta)}")
    print(f"SE_stat:                        {_fmt(res.se_stat)}")
    print(f"SE_mc:                          {_fmt(res.se_mc)}")
    print(f"SE_total:                       {_fmt(res.se_total)}")

    # one-row summary for optional CSV
    row = {
        "file": path.name,
        "span": float(span),
        "n_points": int(len(parameter)),
        "param_min": float(np.min(parameter)),
        "param_max": float(np.max(parameter)),
        "lp_min": float(np.nanmin(loglik)),
        "lp_max": float(np.nanmax(loglik)),
        "smooth_arg_max": float(res.mle),
        "quad_max": float(res.quadratic_max),
        "ci_lower": float(ci_lo) if ci_lo is not None else float("nan"),
        "ci_upper": float(ci_hi) if ci_hi is not None else float("nan"),
        "delta": float(res.delta),
        "SE_stat": float(res.se_stat),
        "SE_mc": float(res.se_mc),
        "SE_total": float(res.se_total),
    }
    return row


def _validate_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not (0 < args.span <= 1.0):
        p.error("--span must be in the interval (0, 1].")
    if not (0.0 < args.level < 1.0):
        p.error("--level must be in the interval (0, 1).")
    if args.n_grid < 3:
        p.error("--n-grid must be >= 3.")
    if not args.file.exists():
        p.error(f"--file does not exist: {args.file}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run MCAP on a single profile CSV using mcap.py")
    p.add_argument("--file", "-f", type=Path, required=True,
                   help="Path to the profile-likelihood CSV.")
    p.add_argument("--span", type=float, default=0.75,
                   help="LOESS span (default 0.75). Must be in (0, 1].")
    p.add_argument("--level", type=float, default=0.95,
                   help="Confidence level (default 0.95). Must be in (0, 1).")
    p.add_argument("--n-grid", type=int, default=1000,
                   help="Grid points for smoothing (default 1000, must be >= 3).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write a one-row CSV summary.")
    args = p.parse_args(argv)

    _validate_args(p, args)

    row = run_one(args.file, span=args.span, level=args.level, n_grid=args.n_grid)

    if args.out:
        pd.DataFrame([row]).to_csv(args.out, index=False)
        print(f"\nSaved summary: {args.out.resolve()}")


if __name__ == "__main__":
    main()