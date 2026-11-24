from __future__ import annotations

import pandas as pd

from prdm9_pipeline import nbcompat


def test_defaults_without_environment(monkeypatch):
    for var in ("PRDM9_THREADS", "PRDM9_MC_ITERS", "PRDM9_N_BOOT"):
        monkeypatch.delenv(var, raising=False)
    assert nbcompat.threads(64) == 64
    assert nbcompat.mc_iters(200_000) == 200_000
    assert nbcompat.n_boot(5000) == 5000


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("PRDM9_THREADS", "8")
    monkeypatch.setenv("PRDM9_MC_ITERS", "1000")
    monkeypatch.setenv("PRDM9_N_BOOT", "10")
    assert nbcompat.threads(64) == 8
    assert nbcompat.mc_iters(200_000) == 1000
    assert nbcompat.n_boot(5000) == 10


def test_display_prints_frames(capsys):
    nbcompat.display(pd.DataFrame({"a": range(100)}), max_rows=3)
    out = capsys.readouterr().out
    assert "a" in out and len(out.strip().splitlines()) == 4  # header + 3 rows
