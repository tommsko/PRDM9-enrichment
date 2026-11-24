"""Shared fixtures. Tests use small synthetic inputs only; no downloaded data are needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def tiny_genome(tmp_path):
    """Two-chromosome FASTA (+ .fai built by pyfaidx on first use)."""
    rng = np.random.default_rng(7)
    seqs = {"chr1": "".join(rng.choice(list("ACGT"), 5000)), "chr2": "".join(rng.choice(list("ACGT"), 3000))}
    fa = tmp_path / "tiny.fa"
    with fa.open("w") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    return fa, seqs


def bed(rows):
    return pd.DataFrame(rows, columns=["Chromosome", "Start", "End"])
