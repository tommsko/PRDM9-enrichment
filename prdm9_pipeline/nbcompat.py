"""Small runtime shims that let the converted notebooks run as plain scripts.

Every value can be overridden through environment variables, which the Snakemake workflow sets
from config/config.yaml:

  PRDM9_THREADS      worker processes per stage (default: the notebook's own value)
  PRDM9_MC_ITERS     Monte Carlo iterations (default 200,000 as in the manuscript; lower only for smoke runs)
  PRDM9_N_BOOT       bootstrap replicates in 3b (default 5,000)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # stages run headless; plt.show() becomes a no-op


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else int(default)


def threads(default: int) -> int:
    return _env_int("PRDM9_THREADS", default)


def mc_iters(default: int) -> int:
    return _env_int("PRDM9_MC_ITERS", default)


def n_boot(default: int) -> int:
    return _env_int("PRDM9_N_BOOT", default)


def display(obj, max_rows: int = 40) -> None:
    """Notebook display() replacement: print DataFrames/objects to the stage log."""
    try:
        import pandas as pd

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.head(max_rows).to_string())
            return
    except ImportError:  # pragma: no cover
        pass
    print(obj)
