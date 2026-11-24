#!/usr/bin/env python3
"""Convert the production notebooks (notebooks/*.ipynb) into runnable stage scripts.

The converted scripts in prdm9_pipeline/stages/ are what the Snakemake workflow executes. The
conversion is deliberately mechanical so that every difference from the published notebooks is
listed here and can be audited:

  1. markdown cells become comments, code cells are concatenated in order (`# %%` markers);
  2. `!pip install ...` lines are removed (dependencies come from envs/prdm9.yml);
  3. IPython's `display` is replaced by `prdm9_pipeline.nbcompat.display` (prints to the log);
  4. worker counts, Monte Carlo iterations and bootstrap replicates read optional overrides from
     the environment (PRDM9_THREADS / PRDM9_MC_ITERS / PRDM9_N_BOOT); defaults are the notebook
     values, so an un-configured run reproduces the published settings.

Paths are NOT rewritten: every stage runs with its working directory set to the results folder,
which contains symlinks to the input data (see workflow/Snakefile), exactly mirroring the layout
the notebooks were developed in.

Usage:  python tools/convert_notebooks.py            (re-generates prdm9_pipeline/stages/*.py)
        python tools/convert_notebooks.py --check    (exit 1 if the stage files are out of date)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
OUT_DIR = REPO / "prdm9_pipeline" / "stages"

STAGES = {
    "1_sample_windows": "s1_sample_windows",
    "2a_feat_G4": "s2a_feat_g4",
    "2b_feat_methylation": "s2b_feat_methylation",
    "2c_feat_motifs": "s2c_feat_motifs",
    "2d_feat_accessibility": "s2d_feat_accessibility",
    "2e_feat_recombination": "s2e_feat_recombination",
    "2f_feat_histones": "s2f_feat_histones",
    "2g_combine": "s2g_combine",
    "3a_enrichment_g4": "s3a_enrichment_g4",
    "3b_enrichment_g4seq": "s3b_enrichment_g4seq",
    "3c_enrichment_all": "s3c_enrichment_all",
}

PROLOGUE = '''"""Stage {stage}: converted from notebooks/{nb}.ipynb by tools/convert_notebooks.py.

Run through the workflow (it sets the working directory and environment); see that converter for the
complete list of differences from the notebook. Edit the notebook + converter, not this file.
"""
from prdm9_pipeline.nbcompat import display, mc_iters, n_boot, threads  # noqa: F401
'''

# (pattern, replacement, expected number of matches) applied with re.MULTILINE to every notebook.
GLOBAL_RULES = [
    (r"^[ \t]*!pip install .*\n", "", None),
    (r"^from IPython\.display import display\n", "# display() is provided by prdm9_pipeline.nbcompat\n", None),
    (r"^(N_CPUS|N_WORKERS|N_CORES|N_JOBS) = ([^#\n]+?)(\s*#.*)?$", r"\1 = threads(\2)\3", 1),
    (r"^N_ITERS = ([^#\n]+?)(\s*#.*)?$", r"N_ITERS = mc_iters(\1)\2", None),
    (r"^N_BOOTSTRAP = ([^#\n]+?)(\s*#.*)?$", r"N_BOOTSTRAP = n_boot(\1)\2", None),
]

def cell_text(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def convert(nb_name: str) -> str:
    nb = json.loads((NB_DIR / f"{nb_name}.ipynb").read_text())
    parts = []
    for i, cell in enumerate(nb["cells"]):
        text = cell_text(cell).rstrip("\n")
        if not text.strip():
            continue
        if cell["cell_type"] == "markdown":
            parts.append(f"# %% [markdown] cell {i}\n" + "\n".join("# " + line if line else "#" for line in text.splitlines()))
        elif cell["cell_type"] == "code":
            parts.append(f"# %% cell {i}\n{text}")
    body = "\n\n".join(parts) + "\n"

    for pattern, repl, expected in GLOBAL_RULES:
        body, n = re.subn(pattern, repl, body, flags=re.MULTILINE)
        if expected is not None and n != expected:
            raise SystemExit(f"{nb_name}: rule {pattern!r} matched {n}x, expected {expected}")
    leftovers = [l for l in body.splitlines() if re.match(r"\s*[!%]", l)]
    if leftovers:
        raise SystemExit(f"{nb_name}: notebook magics left: {leftovers[:3]}")
    # a script (unlike a notebook cell) needs __future__ imports at the very top
    futures = sorted(set(re.findall(r"^from __future__ import .+$", body, flags=re.MULTILINE)))
    body = re.sub(r"^from __future__ import .+\n", "", body, flags=re.MULTILINE)
    prologue = PROLOGUE.format(stage=STAGES[nb_name], nb=nb_name)
    if futures:
        doc_end = prologue.index('"""\n', 3) + 4
        prologue = prologue[:doc_end] + "\n".join(futures) + "\n" + prologue[doc_end:]
    return prologue + "\n" + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only verify that generated files are up to date")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for nb_name, module in STAGES.items():
        text = convert(nb_name)
        target = OUT_DIR / f"{module}.py"
        if args.check:
            if not target.exists() or target.read_text() != text:
                stale.append(target.name)
        else:
            target.write_text(text)
            print(f"wrote {target.relative_to(REPO)}")
    if args.check:
        if stale:
            print("out of date:", ", ".join(stale))
            return 1
        print("stage scripts are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
