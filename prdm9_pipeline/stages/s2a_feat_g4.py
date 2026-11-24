"""Stage s2a_feat_g4: converted from notebooks/2a_feat_G4.ipynb by tools/convert_notebooks.py.

Run through the workflow (it sets the working directory and environment); see that converter for the
complete list of differences from the notebook. Edit the notebook + converter, not this file.
"""
from prdm9_pipeline.nbcompat import display, mc_iters, n_boot, threads  # noqa: F401

# %% [markdown] cell 0
#
# # PRDM9 G-quadruplex feature generation — production notebook
#
# This notebook rebuilds the G4 feature-generation step for publication use.
#
# It is designed to operate on the three window sets produced in the previous sampling notebook:
#
# - **PRDM9 windows**
# - **`datasetA_matched`** (length + GC% + entropy matched background)
# - **`datasetB_global`** (length-matched only background)
#
# ## Feature schema
#
# ### Experimental G4-seq features
# Using **merged** intervals from `g4/HEK293_G4s_hg38.bed`:
# - `g4exp__hits`
# - `g4exp__covfrac`
#
# ### Sequence-derived canonical G4 features
# Primary objects are the canonical intervals loaded from:
# - `g4/G4s_short_pqsfinder_hg38.txt`
# - `g4/G4s_median_pqsfinder_hg38.txt`
# - `g4/G4s_long_pqsfinder_hg38.txt`
#
# Each canonical interval is annotated with:
# - **size class**: `short`, `median`, `long`
# - **score bin**: `Q1`, `Q2`, `Q3`, `Q4`, or `UNK`
# - **experimental support**: `supported`, `unsupported`
#
# Scores are transferred from `g4/pqsfinder_hg38.bed` by a **deterministic fuzzy one-to-one matching** procedure.
# The canonical TXT intervals remain the primary intervals; BED scores are attached only as annotations.
#
# ## Outputs
#
# - detailed feature table with support split
# - collapsed feature table summed over support strata
# - per-window-set split exports
# - score-matching diagnostics tables
# - QC figures and summary tables
# - run metadata JSON
#
# ## Why this differs from the exploratory notebook
#
# The exploratory notebook treated the scored BED intervals as the primary G4 objects and then tried to infer category membership. For publication, that is riskier than necessary. Here, the **TXT intervals define the canonical objects**, while the BED is used only to provide scores. This preserves the intended short/median/long definitions exactly.

# %% cell 1

# Core setup
import os
import gc
import json
import math
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pyranges as pr

from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()

# ------------------------------------------------------------------
# User-tunable configuration
# ------------------------------------------------------------------
N_CPUS = threads(64)
SLACK_BP = 3
SCORE_Q = 4

SAMPLING_DIR = Path("data_prdm9_sampling_production")
PRDM9_BED = SAMPLING_DIR / "prdm9_filtered.bed"
DATASETA_BED = SAMPLING_DIR / "datasetA_matched.len_gc_entropy.bed"
DATASETB_BED = SAMPLING_DIR / "datasetB_global.length_only.bed"

G4_SCORE_BED = Path("g4/pqsfinder_hg38.bed")
G4_SHORT_TXT = Path("g4/G4s_short_pqsfinder_hg38.txt")
G4_MEDIAN_TXT = Path("g4/G4s_median_pqsfinder_hg38.txt")
G4_LONG_TXT = Path("g4/G4s_long_pqsfinder_hg38.txt")
G4_EXPERIMENTAL_BED = Path("g4/HEK293_G4s_hg38.bed")

OUTDIR = Path("data_prdm9_g4_production")
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"
FEATDIR = OUTDIR / "features"
METADIR = OUTDIR / "metadata"
LOGDIR = OUTDIR / "logs"
for p in [OUTDIR, FIGDIR, TABDIR, FEATDIR, METADIR, LOGDIR]:
    p.mkdir(parents=True, exist_ok=True)

DETAILED_PARQUET = FEATDIR / "g4_features_detailed.parquet"
COLLAPSED_PARQUET = FEATDIR / "g4_features_collapsed.parquet"
WINDOWS_TABLE = TABDIR / "all_windows.tsv"

# stable palettes
PALETTE_WINDOWSET = {
    "prdm9": "#C44E52",
    "datasetA_matched": "#4C72B0",
    "datasetB_global": "#55A868",
}
PALETTE_SIZE = {
    "short": "#4C72B0",
    "median": "#DD8452",
    "long": "#C44E52",
}
PALETTE_Q = {
    "Q1": "#B2BEB5",
    "Q2": "#7A9CC6",
    "Q3": "#4F77B6",
    "Q4": "#1F3E75",
    "UNK": "#7F7F7F",
}

# logging
logger = logging.getLogger("prdm9_g4")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

file_handler = logging.FileHandler(LOGDIR / "prdm9_g4_feature_generation.log", mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

console.print(Panel.fit(
    f"[bold]PRDM9 G4 production feature notebook[/bold]\n"
    f"N_CPUS = {N_CPUS}\n"
    f"SLACK_BP = {SLACK_BP}\n"
    f"SCORE_Q = {SCORE_Q}\n"
    f"Output directory: {OUTDIR}",
    border_style="cyan"
))

# %% cell 2

# Helper functions

def read_bed3(path, extra_cols=None, names=None):
    if extra_cols is None:
        extra_cols = []
    if names is None:
        names = ["Chromosome", "Start", "End"] + list(extra_cols)
    usecols = list(range(len(names)))
    df = pd.read_csv(path, sep="	", header=None, usecols=usecols, names=names)
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    return df


def read_score_bed(path):
    df = pd.read_csv(
        path,
        sep="	",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["Chromosome", "Start", "End", "Name", "Score", "Strand"],
    )
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Strand"] = df["Strand"].astype(str)
    return df


def read_canonical_txt(path, size_class):
    df = pd.read_csv(path, sep="	", header=0)
    rename_map = {"seqnames": "Chromosome", "start": "Start", "end": "End"}
    df = df.rename(columns=rename_map)
    required = ["Chromosome", "Start", "End"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df[["Chromosome", "Start", "End"]].copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    df["SizeClass"] = size_class
    return df


def assert_valid_intervals(df, name):
    if (df["End"] <= df["Start"]).any():
        bad = df.loc[df["End"] <= df["Start"]].head()
        raise ValueError(f"{name} contains non-positive intervals. Example:\\n{bad}")


def assign_rank_bins(values, n_bins=4):
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return np.array([], dtype=object)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(vals), dtype=np.int64)
    pct = ranks / max(len(vals), 1)
    bins = np.floor(pct * n_bins).astype(np.int64) + 1
    bins = np.clip(bins, 1, n_bins)
    return np.array([f"Q{x}" for x in bins], dtype=object)


def make_detailed_feature_name(kind, size_class=None, support=None, qbin=None):
    if kind == "exp":
        return f"g4exp__{{stat}}"
    sup = "supported" if bool(support) else "unsupported"
    return f"g4seq__{size_class}__{sup}__{qbin}__{{stat}}"


def make_collapsed_feature_name(kind, size_class=None, qbin=None):
    if kind == "exp":
        return f"g4exp__{{stat}}"
    return f"g4seq_collapsed__{size_class}__{qbin}__{{stat}}"


def reciprocal_overlap(a_start, a_end, b_start, b_end):
    ov = np.maximum(0, np.minimum(a_end, b_end) - np.maximum(a_start, b_start))
    al = np.maximum(1, a_end - a_start)
    bl = np.maximum(1, b_end - b_start)
    return ov / al, ov / bl, ov


def chromosome_key(chrom):
    c = chrom.replace("chr", "")
    try:
        return (0, int(c))
    except ValueError:
        order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
        return (1, order.get(c, 10_000), c)


def save_table(df, path):
    df.to_csv(path, sep="	", index=False)
    logger.info(f"[write] {path} ({len(df):,} rows)")


def write_split_feature_tables(df, stem):
    for window_set, sub in df.groupby("WindowSet", sort=False):
        out = FEATDIR / f"{window_set}.{stem}.parquet"
        sub.to_parquet(out, index=False)
        logger.info(f"[write] {out} ({len(sub):,} rows)")


def add_global_window_id(df):
    df = df.copy()
    df["window_id"] = (
        df["WindowSet"].astype(str)
        + "::"
        + df["Chromosome"].astype(str)
        + ":"
        + df["Start"].astype(str)
        + "-"
        + df["End"].astype(str)
    )
    return df


def summarize_numeric(series):
    s = pd.Series(series)
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }

# %% cell 3

# Load windows from the previous notebook
logger.info("[1/8] Loading window definitions ...")

prdm9 = read_bed3(PRDM9_BED)
prdm9["WindowSet"] = "prdm9"

datasetA = read_bed3(DATASETA_BED)
datasetA["WindowSet"] = "datasetA_matched"

datasetB = read_bed3(DATASETB_BED)
datasetB["WindowSet"] = "datasetB_global"

windows = pd.concat([prdm9, datasetA, datasetB], ignore_index=True)
windows = add_global_window_id(windows)
windows["window_len"] = (windows["End"] - windows["Start"]).astype(np.int64)
windows = windows.sort_values(["WindowSet", "Chromosome", "Start", "End"], key=lambda x: x.map(chromosome_key) if x.name == "Chromosome" else x).reset_index(drop=True)

assert_valid_intervals(windows, "windows")
if windows["window_id"].duplicated().any():
    dup = windows.loc[windows["window_id"].duplicated(keep=False)].head()
    raise ValueError(f"Duplicate window_id values detected. Example:\\n{dup}")

save_table(windows[["WindowSet", "Chromosome", "Start", "End", "window_id", "window_len"]], WINDOWS_TABLE)

t = Table(title="Window summary")
t.add_column("WindowSet")
t.add_column("N", justify="right")
t.add_column("Median length", justify="right")
for ws, sub in windows.groupby("WindowSet", sort=False):
    t.add_row(ws, f"{len(sub):,}", f"{int(sub['window_len'].median()):,}")
console.print(t)

# %% cell 4

# Load G4 source annotations
logger.info("[2/8] Loading G4 source annotations ...")

score_bed = read_score_bed(G4_SCORE_BED)
assert_valid_intervals(score_bed, "pqsfinder score bed")
score_bed["score_id"] = np.arange(len(score_bed), dtype=np.int64)
score_bed["score_len"] = (score_bed["End"] - score_bed["Start"]).astype(np.int64)

short_txt = read_canonical_txt(G4_SHORT_TXT, "short")
median_txt = read_canonical_txt(G4_MEDIAN_TXT, "median")
long_txt = read_canonical_txt(G4_LONG_TXT, "long")
canonical = pd.concat([short_txt, median_txt, long_txt], ignore_index=True)
assert_valid_intervals(canonical, "canonical txt intervals")
canonical["seq_id"] = np.arange(len(canonical), dtype=np.int64)
canonical["seq_len"] = (canonical["End"] - canonical["Start"]).astype(np.int64)

experimental_raw = read_bed3(G4_EXPERIMENTAL_BED)
assert_valid_intervals(experimental_raw, "experimental G4 bed")
experimental_merged = pr.PyRanges(experimental_raw).merge().df
experimental_merged["exp_id"] = np.arange(len(experimental_merged), dtype=np.int64)
experimental_merged["exp_len"] = (experimental_merged["End"] - experimental_merged["Start"]).astype(np.int64)

logger.info(
    f"pqsfinder scored intervals: {len(score_bed):,} | "
    f"canonical txt intervals: {len(canonical):,} | "
    f"experimental merged intervals: {len(experimental_merged):,}"
)

src_summary = pd.DataFrame([
    {"Source": "pqsfinder_score_bed", "N": len(score_bed)},
    {"Source": "canonical_short_txt", "N": len(short_txt)},
    {"Source": "canonical_median_txt", "N": len(median_txt)},
    {"Source": "canonical_long_txt", "N": len(long_txt)},
    {"Source": "canonical_total_txt", "N": len(canonical)},
    {"Source": "experimental_raw", "N": len(experimental_raw)},
    {"Source": "experimental_merged", "N": len(experimental_merged)},
])
save_table(src_summary, TABDIR / "g4_source_summary.tsv")

# %% cell 5

# Deterministic fuzzy matching: canonical TXT intervals -> score BED
logger.info("[3/8] Fuzzy one-to-one matching of canonical TXT intervals to pqsfinder scores ...")

seq_join = canonical[["Chromosome", "Start", "End", "seq_id", "SizeClass", "seq_len"]].copy()
seq_join = seq_join.rename(columns={"Start": "orig_Start", "End": "orig_End"})
seq_join["Start"] = (seq_join["orig_Start"] - SLACK_BP).clip(lower=0).astype(np.int64)
seq_join["End"] = (seq_join["orig_End"] + SLACK_BP).astype(np.int64)

candidates = pr.PyRanges(seq_join).join(
    pr.PyRanges(score_bed[["Chromosome", "Start", "End", "score_id", "Score", "score_len"]]),
    suffix="_score"
).df

if candidates.empty:
    raise RuntimeError("No candidate matches found between canonical TXT intervals and pqsfinder score BED.")

candidates["abs_start_diff"] = (candidates["orig_Start"] - candidates["Start_score"]).abs().astype(np.int64)
candidates["abs_end_diff"] = (candidates["orig_End"] - candidates["End_score"]).abs().astype(np.int64)
candidates["sum_delta"] = (candidates["abs_start_diff"] + candidates["abs_end_diff"]).astype(np.int64)
candidates["len_delta"] = ((candidates["orig_End"] - candidates["orig_Start"]) - (candidates["End_score"] - candidates["Start_score"])).abs().astype(np.int64)
ro1, ro2, ov = reciprocal_overlap(
    candidates["orig_Start"].values,
    candidates["orig_End"].values,
    candidates["Start_score"].values,
    candidates["End_score"].values,
)
candidates["overlap_bp"] = ov.astype(np.int64)
candidates["recip_seq"] = ro1.astype(np.float64)
candidates["recip_score"] = ro2.astype(np.float64)

# Strict acceptance rule: both ends within slack and some true overlap on original coordinates
candidates = candidates[
    (candidates["abs_start_diff"] <= SLACK_BP)
    & (candidates["abs_end_diff"] <= SLACK_BP)
    & (candidates["overlap_bp"] > 0)
].copy()

if candidates.empty:
    raise RuntimeError("All candidate score matches were filtered out by the slack rule.")

# Greedy deterministic one-to-one assignment over globally sorted candidate pairs
candidates = candidates.sort_values(
    ["sum_delta", "len_delta", "abs_start_diff", "abs_end_diff", "overlap_bp", "score_id", "seq_id"],
    ascending=[True, True, True, True, False, True, True],
).reset_index(drop=True)

used_seq = set()
used_score = set()
chosen_rows = []
for row in tqdm(candidates.itertuples(index=False), total=len(candidates), desc="matching canonical->score"):
    if row.seq_id in used_seq or row.score_id in used_score:
        continue
    chosen_rows.append(row)
    used_seq.add(row.seq_id)
    used_score.add(row.score_id)

match_df = pd.DataFrame(chosen_rows)
if match_df.empty:
    raise RuntimeError("No one-to-one canonical->score matches survived greedy assignment.")

if match_df["seq_id"].duplicated().any() or match_df["score_id"].duplicated().any():
    raise AssertionError("One-to-one score matching failed uniqueness check.")

canonical_annot = canonical.merge(
    match_df[[
        "seq_id", "score_id", "Score", "Start_score", "End_score",
        "abs_start_diff", "abs_end_diff", "sum_delta", "len_delta",
        "overlap_bp", "recip_seq", "recip_score"
    ]],
    on="seq_id",
    how="left",
)
canonical_annot["score_matched"] = canonical_annot["score_id"].notna()

matched_mask = canonical_annot["score_matched"].values
canonical_annot["QBin"] = "UNK"
canonical_annot.loc[matched_mask, "QBin"] = assign_rank_bins(canonical_annot.loc[matched_mask, "Score"].values, n_bins=SCORE_Q)

match_summary = (
    canonical_annot
    .groupby(["SizeClass", "score_matched"], as_index=False)
    .size()
    .rename(columns={"size": "N"})
)
match_summary["Fraction_within_size"] = match_summary.groupby("SizeClass")["N"].transform(lambda s: s / s.sum())
save_table(match_summary, TABDIR / "g4_score_match_summary.tsv")

match_detail = canonical_annot[[
    "seq_id", "Chromosome", "Start", "End", "seq_len", "SizeClass",
    "score_id", "Score", "Start_score", "End_score",
    "score_matched", "QBin",
    "abs_start_diff", "abs_end_diff", "sum_delta", "len_delta",
    "overlap_bp", "recip_seq", "recip_score"
]].copy()
save_table(match_detail, TABDIR / "g4_score_match_detail.tsv")

logger.info(
    f"Matched canonical intervals: {int(canonical_annot['score_matched'].sum()):,} / {len(canonical_annot):,} "
    f"({canonical_annot['score_matched'].mean():.2%})"
)

# %% cell 6

# Experimental support annotation on canonical TXT intervals
logger.info("[4/8] Annotating canonical intervals as supported/unsupported using merged experimental G4s ...")

support_hits = pr.PyRanges(canonical_annot[["Chromosome", "Start", "End", "seq_id"]]).join(
    pr.PyRanges(experimental_merged[["Chromosome", "Start", "End", "exp_id"]]),
    suffix="_exp"
).df

supported_ids = np.array([], dtype=np.int64) if support_hits.empty else np.sort(support_hits["seq_id"].unique())
canonical_annot["Supported"] = False
canonical_annot.loc[canonical_annot["seq_id"].isin(supported_ids), "Supported"] = True

support_summary = (
    canonical_annot
    .groupby(["SizeClass", "QBin", "Supported"], as_index=False)
    .size()
    .rename(columns={"size": "N"})
)
support_summary["Fraction_within_size_qbin"] = support_summary.groupby(["SizeClass", "QBin"])["N"].transform(lambda s: s / s.sum())
save_table(support_summary, TABDIR / "g4_support_summary.tsv")

canonical_out = canonical_annot[[
    "seq_id", "Chromosome", "Start", "End", "seq_len", "SizeClass",
    "score_matched", "Score", "QBin", "Supported",
    "score_id", "Start_score", "End_score",
    "abs_start_diff", "abs_end_diff", "sum_delta", "len_delta",
    "overlap_bp", "recip_seq", "recip_score"
]].copy()
save_table(canonical_out, TABDIR / "canonical_g4_annotations.tsv")

annot_summary = Table(title="Canonical G4 annotation summary")
annot_summary.add_column("Metric")
annot_summary.add_column("Value", justify="right")
annot_summary.add_row("Canonical intervals", f"{len(canonical_annot):,}")
annot_summary.add_row("Score matched", f"{int(canonical_annot['score_matched'].sum()):,}")
annot_summary.add_row("Unknown score bin (UNK)", f"{int((canonical_annot['QBin'] == 'UNK').sum()):,}")
annot_summary.add_row("Supported", f"{int(canonical_annot['Supported'].sum()):,}")
annot_summary.add_row("Unsupported", f"{int((~canonical_annot['Supported']).sum()):,}")
console.print(annot_summary)

# %% cell 7

# Prepare expected feature schemas
logger.info("[5/8] Building expected feature schemas ...")

SIZE_CLASSES = ["short", "median", "long"]
QBINS = ["Q1", "Q2", "Q3", "Q4", "UNK"]
SUPPORT_STATES = [True, False]

expected_detailed = []
expected_collapsed = []
for stat in ["hits", "covfrac"]:
    expected_detailed.append(f"g4exp__{stat}")
    expected_collapsed.append(f"g4exp__{stat}")

for size_class in SIZE_CLASSES:
    for supported in SUPPORT_STATES:
        for qbin in QBINS:
            expected_detailed.append(make_detailed_feature_name("seq", size_class, supported, qbin).format(stat="hits"))
            expected_detailed.append(make_detailed_feature_name("seq", size_class, supported, qbin).format(stat="covfrac"))

for size_class in SIZE_CLASSES:
    for qbin in QBINS:
        expected_collapsed.append(make_collapsed_feature_name("seq", size_class, qbin).format(stat="hits"))
        expected_collapsed.append(make_collapsed_feature_name("seq", size_class, qbin).format(stat="covfrac"))

logger.info(f"Detailed feature count: {len(expected_detailed)} (expected 62)")
logger.info(f"Collapsed feature count: {len(expected_collapsed)} (expected 32)")
assert len(expected_detailed) == 62
assert len(expected_collapsed) == 32

# %% cell 8

# Parallel feature generation by (window set, chromosome)
logger.info("[6/8] Generating window-level G4 features in parallel ...")

windows_by_chunk = {}
for (window_set, chrom), sub in windows.groupby(["WindowSet", "Chromosome"], sort=False):
    windows_by_chunk[(window_set, chrom)] = sub[["WindowSet", "Chromosome", "Start", "End", "window_id", "window_len"]].copy()

exp_by_chrom = {
    chrom: sub[["Chromosome", "Start", "End", "exp_id"]].copy()
    for chrom, sub in experimental_merged.groupby("Chromosome", sort=False)
}
seq_by_chrom = {
    chrom: sub[["Chromosome", "Start", "End", "seq_id", "SizeClass", "Supported", "QBin"]].copy()
    for chrom, sub in canonical_annot.groupby("Chromosome", sort=False)
}


def process_window_chunk(window_set, chrom, win_df):
    base = win_df[["WindowSet", "Chromosome", "Start", "End", "window_id", "window_len"]].copy()

    # start with all windows present
    out = base[["WindowSet", "Chromosome", "Start", "End", "window_id"]].copy()

    # --------------------------
    # Experimental G4 features
    # --------------------------
    exp_chr = exp_by_chrom.get(chrom)
    if exp_chr is None or exp_chr.empty:
        out["g4exp__hits"] = 0
        out["g4exp__covfrac"] = 0.0
    else:
        join_exp = pr.PyRanges(base[["Chromosome", "Start", "End", "window_id"]]).join(
            pr.PyRanges(exp_chr),
            suffix="_exp"
        ).df
        if join_exp.empty:
            out["g4exp__hits"] = 0
            out["g4exp__covfrac"] = 0.0
        else:
            join_exp["int_start"] = np.maximum(join_exp["Start"].values, join_exp["Start_exp"].values)
            join_exp["int_end"] = np.minimum(join_exp["End"].values, join_exp["End_exp"].values)
            join_exp["int_len"] = (join_exp["int_end"] - join_exp["int_start"]).astype(np.int64)
            join_exp = join_exp[join_exp["int_len"] > 0].copy()

            exp_hits = join_exp.groupby("window_id", as_index=False)["exp_id"].nunique().rename(columns={"exp_id": "g4exp__hits"})
            exp_cov = join_exp.groupby("window_id", as_index=False)["int_len"].sum().rename(columns={"int_len": "covered_bp"})
            exp_cov = exp_cov.merge(base[["window_id", "window_len"]], on="window_id", how="left")
            exp_cov["g4exp__covfrac"] = (exp_cov["covered_bp"] / exp_cov["window_len"]).astype(np.float32)
            exp_cov = exp_cov[["window_id", "g4exp__covfrac"]]

            out = out.merge(exp_hits, on="window_id", how="left")
            out = out.merge(exp_cov, on="window_id", how="left")
            out["g4exp__hits"] = out["g4exp__hits"].fillna(0).astype(np.int32)
            out["g4exp__covfrac"] = out["g4exp__covfrac"].fillna(0.0).astype(np.float32)

    # --------------------------
    # Sequence-derived detailed features
    # --------------------------
    seq_chr = seq_by_chrom.get(chrom)
    if seq_chr is None or seq_chr.empty:
        for col in expected_detailed:
            if col.startswith("g4seq__"):
                out[col] = 0
        detailed = out.copy()
        collapsed = out[["WindowSet", "Chromosome", "Start", "End", "window_id", "g4exp__hits", "g4exp__covfrac"]].copy()
        for col in expected_collapsed:
            if col.startswith("g4seq_collapsed__"):
                collapsed[col] = 0
        return detailed, collapsed

    join_seq = pr.PyRanges(base[["Chromosome", "Start", "End", "window_id"]]).join(
        pr.PyRanges(seq_chr),
        suffix="_seq"
    ).df

    if join_seq.empty:
        detailed = out.copy()
        for col in expected_detailed:
            if col.startswith("g4seq__") and col not in detailed.columns:
                detailed[col] = 0
        collapsed = out[["WindowSet", "Chromosome", "Start", "End", "window_id", "g4exp__hits", "g4exp__covfrac"]].copy()
        for col in expected_collapsed:
            if col.startswith("g4seq_collapsed__") and col not in collapsed.columns:
                collapsed[col] = 0
        return detailed, collapsed

    join_seq["int_start"] = np.maximum(join_seq["Start"].values, join_seq["Start_seq"].values)
    join_seq["int_end"] = np.minimum(join_seq["End"].values, join_seq["End_seq"].values)
    join_seq["int_len"] = (join_seq["int_end"] - join_seq["int_start"]).astype(np.int64)
    join_seq = join_seq[join_seq["int_len"] > 0].copy()

    # hits: exact because each seq interval belongs to exactly one detailed group
    hits_det = (
        join_seq.groupby(["window_id", "SizeClass", "Supported", "QBin"], as_index=False)["seq_id"]
        .nunique()
        .rename(columns={"seq_id": "hits"})
    )
    hits_det["feature"] = [
        make_detailed_feature_name("seq", s, sup, q).format(stat="hits")
        for s, sup, q in zip(hits_det["SizeClass"], hits_det["Supported"], hits_det["QBin"])
    ]
    hits_det_wide = hits_det.pivot_table(index="window_id", columns="feature", values="hits", aggfunc="sum", fill_value=0)

    # detailed covfrac: merge intersection segments within each group to avoid double counting
    seg_det = join_seq[["Chromosome", "int_start", "int_end", "window_id", "SizeClass", "Supported", "QBin"]].copy()
    seg_det = seg_det.rename(columns={"int_start": "Start", "int_end": "End"})
    merged_det = pr.PyRanges(seg_det).merge(by=["window_id", "SizeClass", "Supported", "QBin"]).df
    merged_det["seg_len"] = (merged_det["End"] - merged_det["Start"]).astype(np.int64)
    cov_det = merged_det.groupby(["window_id", "SizeClass", "Supported", "QBin"], as_index=False)["seg_len"].sum().rename(columns={"seg_len": "covered_bp"})
    cov_det = cov_det.merge(base[["window_id", "window_len"]], on="window_id", how="left")
    cov_det["covfrac"] = (cov_det["covered_bp"] / cov_det["window_len"]).astype(np.float32)
    cov_det["feature"] = [
        make_detailed_feature_name("seq", s, sup, q).format(stat="covfrac")
        for s, sup, q in zip(cov_det["SizeClass"], cov_det["Supported"], cov_det["QBin"])
    ]
    cov_det_wide = cov_det.pivot_table(index="window_id", columns="feature", values="covfrac", aggfunc="sum", fill_value=0.0)

    detailed = out.copy().set_index("window_id")
    detailed = detailed.join(hits_det_wide, how="left").join(cov_det_wide, how="left")
    detailed = detailed.fillna(0).reset_index()
    for col in expected_detailed:
        if col not in detailed.columns:
            detailed[col] = 0
    detailed = detailed[["WindowSet", "Chromosome", "Start", "End", "window_id"] + expected_detailed]

    # --------------------------
    # Collapsed-over-support features
    # --------------------------
    hits_col = (
        join_seq.groupby(["window_id", "SizeClass", "QBin"], as_index=False)["seq_id"]
        .nunique()
        .rename(columns={"seq_id": "hits"})
    )
    hits_col["feature"] = [
        make_collapsed_feature_name("seq", s, q).format(stat="hits")
        for s, q in zip(hits_col["SizeClass"], hits_col["QBin"])
    ]
    hits_col_wide = hits_col.pivot_table(index="window_id", columns="feature", values="hits", aggfunc="sum", fill_value=0)

    seg_col = join_seq[["Chromosome", "int_start", "int_end", "window_id", "SizeClass", "QBin"]].copy()
    seg_col = seg_col.rename(columns={"int_start": "Start", "int_end": "End"})
    merged_col = pr.PyRanges(seg_col).merge(by=["window_id", "SizeClass", "QBin"]).df
    merged_col["seg_len"] = (merged_col["End"] - merged_col["Start"]).astype(np.int64)
    cov_col = merged_col.groupby(["window_id", "SizeClass", "QBin"], as_index=False)["seg_len"].sum().rename(columns={"seg_len": "covered_bp"})
    cov_col = cov_col.merge(base[["window_id", "window_len"]], on="window_id", how="left")
    cov_col["covfrac"] = (cov_col["covered_bp"] / cov_col["window_len"]).astype(np.float32)
    cov_col["feature"] = [
        make_collapsed_feature_name("seq", s, q).format(stat="covfrac")
        for s, q in zip(cov_col["SizeClass"], cov_col["QBin"])
    ]
    cov_col_wide = cov_col.pivot_table(index="window_id", columns="feature", values="covfrac", aggfunc="sum", fill_value=0.0)

    collapsed = out.copy().set_index("window_id")
    collapsed = collapsed.join(hits_col_wide, how="left").join(cov_col_wide, how="left")
    collapsed = collapsed.fillna(0).reset_index()
    for col in expected_collapsed:
        if col not in collapsed.columns:
            collapsed[col] = 0
    collapsed = collapsed[["WindowSet", "Chromosome", "Start", "End", "window_id"] + expected_collapsed]

    return detailed, collapsed


chunks = sorted(windows_by_chunk.items(), key=lambda kv: (kv[0][0], chromosome_key(kv[0][1])))
logger.info(f"Processing {len(chunks):,} chunks with up to {N_CPUS} CPUs")

results = Parallel(n_jobs=min(N_CPUS, len(chunks)), backend="loky", verbose=0)(
    delayed(process_window_chunk)(window_set, chrom, win_df)
    for (window_set, chrom), win_df in tqdm(chunks, total=len(chunks), desc="feature chunks")
)

detailed_parts = [x[0] for x in results if x[0] is not None]
collapsed_parts = [x[1] for x in results if x[1] is not None]

detailed_features = pd.concat(detailed_parts, ignore_index=True)
collapsed_features = pd.concat(collapsed_parts, ignore_index=True)

# Reorder consistently with original windows
order_df = windows[["window_id", "WindowSet", "Chromosome", "Start", "End"]].copy()
detailed_features = order_df.merge(detailed_features, on=["window_id", "WindowSet", "Chromosome", "Start", "End"], how="left")
collapsed_features = order_df.merge(collapsed_features, on=["window_id", "WindowSet", "Chromosome", "Start", "End"], how="left")

detailed_features = detailed_features.fillna(0)
collapsed_features = collapsed_features.fillna(0)

# strict row/column checks
assert len(detailed_features) == len(windows)
assert len(collapsed_features) == len(windows)
assert detailed_features["window_id"].is_unique
assert collapsed_features["window_id"].is_unique
assert set(expected_detailed).issubset(detailed_features.columns)
assert set(expected_collapsed).issubset(collapsed_features.columns)
assert not detailed_features[expected_detailed].isna().any().any()
assert not collapsed_features[expected_collapsed].isna().any().any()

# make sure hits are integer-like and coverage fractions are bounded per-group
for col in detailed_features.columns:
    if col.endswith("__hits"):
        detailed_features[col] = detailed_features[col].astype(np.int32)
    elif col.endswith("__covfrac"):
        detailed_features[col] = detailed_features[col].astype(np.float32)
        if ((detailed_features[col] < -1e-9) | (detailed_features[col] > 1 + 1e-9)).any():
            raise ValueError(f"Detailed coverage feature out of [0,1]: {col}")
for col in collapsed_features.columns:
    if col.endswith("__hits"):
        collapsed_features[col] = collapsed_features[col].astype(np.int32)
    elif col.endswith("__covfrac"):
        collapsed_features[col] = collapsed_features[col].astype(np.float32)
        if ((collapsed_features[col] < -1e-9) | (collapsed_features[col] > 1 + 1e-9)).any():
            raise ValueError(f"Collapsed coverage feature out of [0,1]: {col}")

# hits consistency: collapsed hits must equal detailed hits summed over support
for size_class in SIZE_CLASSES:
    for qbin in QBINS:
        ccol = make_collapsed_feature_name("seq", size_class, qbin).format(stat="hits")
        dcols = [
            make_detailed_feature_name("seq", size_class, True, qbin).format(stat="hits"),
            make_detailed_feature_name("seq", size_class, False, qbin).format(stat="hits"),
        ]
        if not np.array_equal(collapsed_features[ccol].values, detailed_features[dcols].sum(axis=1).values):
            raise AssertionError(f"Collapsed hit mismatch for {size_class} {qbin}")

logger.info("Feature generation complete.")
logger.info(f"Detailed matrix shape:  {detailed_features.shape}")
logger.info(f"Collapsed matrix shape: {collapsed_features.shape}")

# %% cell 9

# Save outputs and build summary tables
logger.info("[7/8] Writing feature tables and summary tables ...")

# combined outputs

detailed_features.to_parquet(DETAILED_PARQUET, index=False)
collapsed_features.to_parquet(COLLAPSED_PARQUET, index=False)
logger.info(f"[write] {DETAILED_PARQUET} ({len(detailed_features):,} rows)")
logger.info(f"[write] {COLLAPSED_PARQUET} ({len(collapsed_features):,} rows)")

write_split_feature_tables(detailed_features, "g4_features_detailed")
write_split_feature_tables(collapsed_features, "g4_features_collapsed")

# feature means by window set
feature_mean_rows = []
for feature_table_name, feat_df, feat_cols in [
    ("detailed", detailed_features, expected_detailed),
    ("collapsed", collapsed_features, expected_collapsed),
]:
    for window_set, sub in feat_df.groupby("WindowSet", sort=False):
        means = sub[feat_cols].mean(axis=0)
        medians = sub[feat_cols].median(axis=0)
        nonzero = (sub[feat_cols] > 0).mean(axis=0)
        for feat in feat_cols:
            feature_mean_rows.append({
                "FeatureTable": feature_table_name,
                "WindowSet": window_set,
                "Feature": feat,
                "Mean": float(means[feat]),
                "Median": float(medians[feat]),
                "NonzeroFraction": float(nonzero[feat]),
            })
feature_means = pd.DataFrame(feature_mean_rows)
save_table(feature_means, TABDIR / "g4_feature_windowset_summary.tsv")

# quick PRDM9/background ratios on collapsed features
ratio_rows = []
ratio_features = [c for c in expected_collapsed if c != "g4exp__hits" and c != "g4exp__covfrac"] + ["g4exp__hits", "g4exp__covfrac"]
for feat in ratio_features:
    p = float(feature_means.query("FeatureTable == 'collapsed' and WindowSet == 'prdm9' and Feature == @feat")["Mean"].iloc[0])
    a = float(feature_means.query("FeatureTable == 'collapsed' and WindowSet == 'datasetA_matched' and Feature == @feat")["Mean"].iloc[0])
    b = float(feature_means.query("FeatureTable == 'collapsed' and WindowSet == 'datasetB_global' and Feature == @feat")["Mean"].iloc[0])
    ratio_rows.append({
        "Feature": feat,
        "Mean_PRDM9": p,
        "Mean_datasetA": a,
        "Mean_datasetB": b,
        "Ratio_PRDM9_over_datasetA": (p + 1e-9) / (a + 1e-9),
        "Ratio_PRDM9_over_datasetB": (p + 1e-9) / (b + 1e-9),
        "Log2Ratio_PRDM9_over_datasetA": np.log2((p + 1e-9) / (a + 1e-9)),
        "Log2Ratio_PRDM9_over_datasetB": np.log2((p + 1e-9) / (b + 1e-9)),
    })
ratio_df = pd.DataFrame(ratio_rows).sort_values("Log2Ratio_PRDM9_over_datasetA", ascending=False)
save_table(ratio_df, TABDIR / "g4_feature_mean_ratios.tsv")

# metadata
metadata = {
    "n_cpus": N_CPUS,
    "slack_bp": SLACK_BP,
    "score_bins": SCORE_Q,
    "window_counts": windows.groupby("WindowSet").size().to_dict(),
    "n_score_bed": int(len(score_bed)),
    "n_canonical_txt": int(len(canonical)),
    "n_canonical_matched_score": int(canonical_annot["score_matched"].sum()),
    "fraction_canonical_matched_score": float(canonical_annot["score_matched"].mean()),
    "n_canonical_supported": int(canonical_annot["Supported"].sum()),
    "n_experimental_merged": int(len(experimental_merged)),
    "detailed_feature_count": int(len(expected_detailed)),
    "collapsed_feature_count": int(len(expected_collapsed)),
    "output_files": {
        "detailed_parquet": str(DETAILED_PARQUET),
        "collapsed_parquet": str(COLLAPSED_PARQUET),
    },
}
with open(METADIR / "g4_run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
logger.info(f"[write] {METADIR / 'g4_run_metadata.json'}")

# %% cell 10

# Validation figures
logger.info("[8/8] Creating QC figures ...")

# Validation-only per-window summaries using the collapsed table and the canonical annotations
plot_df = collapsed_features.merge(windows[["window_id", "window_len"]], on="window_id", how="left")
seq_hit_cols = [c for c in collapsed_features.columns if c.startswith("g4seq_collapsed__") and c.endswith("__hits")]
seq_cov_cols = [c for c in collapsed_features.columns if c.startswith("g4seq_collapsed__") and c.endswith("__covfrac")]
plot_df["g4seq_total_hits_sumgroups"] = plot_df[seq_hit_cols].sum(axis=1)
plot_df["g4seq_total_covfrac_sumgroups"] = plot_df[seq_cov_cols].sum(axis=1)

# any-canonical union coverage and hits for validation only
seq_any = canonical_annot[["Chromosome", "Start", "End", "seq_id"]].copy()
any_rows = []
for (window_set, chrom), win_sub in tqdm(windows.groupby(["WindowSet", "Chromosome"], sort=False), desc="validation any-canonical"):
    seq_chr = seq_any.loc[seq_any["Chromosome"] == chrom]
    if seq_chr.empty:
        tmp = win_sub[["window_id"]].copy()
        tmp["g4seq_any_hits"] = 0
        tmp["g4seq_any_covfrac"] = 0.0
        any_rows.append(tmp)
        continue
    j = pr.PyRanges(win_sub[["Chromosome", "Start", "End", "window_id"]]).join(pr.PyRanges(seq_chr), suffix="_seq").df
    if j.empty:
        tmp = win_sub[["window_id"]].copy()
        tmp["g4seq_any_hits"] = 0
        tmp["g4seq_any_covfrac"] = 0.0
        any_rows.append(tmp)
        continue
    j["int_start"] = np.maximum(j["Start"].values, j["Start_seq"].values)
    j["int_end"] = np.minimum(j["End"].values, j["End_seq"].values)
    j["int_len"] = (j["int_end"] - j["int_start"]).astype(np.int64)
    j = j[j["int_len"] > 0].copy()
    hits = j.groupby("window_id", as_index=False)["seq_id"].nunique().rename(columns={"seq_id": "g4seq_any_hits"})
    seg = j[["Chromosome", "int_start", "int_end", "window_id"]].copy().rename(columns={"int_start": "Start", "int_end": "End"})
    merged = pr.PyRanges(seg).merge(by=["window_id"]).df
    merged["seg_len"] = (merged["End"] - merged["Start"]).astype(np.int64)
    cov = merged.groupby("window_id", as_index=False)["seg_len"].sum().rename(columns={"seg_len": "covered_bp"})
    cov = cov.merge(win_sub[["window_id", "window_len"]], on="window_id", how="left")
    cov["g4seq_any_covfrac"] = (cov["covered_bp"] / cov["window_len"]).astype(np.float32)
    tmp = win_sub[["window_id"]].copy()
    tmp = tmp.merge(hits, on="window_id", how="left").merge(cov[["window_id", "g4seq_any_covfrac"]], on="window_id", how="left")
    tmp = tmp.fillna(0)
    any_rows.append(tmp)

any_df = pd.concat(any_rows, ignore_index=True)
plot_df = plot_df.merge(any_df, on="window_id", how="left")
plot_df[["g4seq_any_hits", "g4seq_any_covfrac"]] = plot_df[["g4seq_any_hits", "g4seq_any_covfrac"]].fillna(0)

# -----------------------------
# Figure 1: annotation / score-match QC
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# A: matched vs unknown by size class
ax = axes[0, 0]
ms = (
    canonical_annot.groupby(["SizeClass", "QBin"]).size().unstack(fill_value=0)
    [["Q1", "Q2", "Q3", "Q4", "UNK"]]
)
left = np.zeros(len(ms), dtype=float)
for qbin in ["Q1", "Q2", "Q3", "Q4", "UNK"]:
    vals = ms[qbin].values
    ax.bar(ms.index, vals, bottom=left, color=PALETTE_Q[qbin], edgecolor="white", linewidth=0.8, label=qbin)
    left += vals
ax.set_title("Canonical G4 intervals by size class and score bin")
ax.set_ylabel("Count")
ax.legend(frameon=False, ncol=5, title="Score bin")

# B: support rate heatmap-like matrix
ax = axes[0, 1]
heat = (
    canonical_annot.groupby(["SizeClass", "QBin", "Supported"]).size().rename("N").reset_index()
    .pivot_table(index="SizeClass", columns=["QBin", "Supported"], values="N", fill_value=0)
)
rate = pd.DataFrame(index=SIZE_CLASSES, columns=QBINS, dtype=float)
for s in SIZE_CLASSES:
    for q in QBINS:
        sub = canonical_annot[(canonical_annot["SizeClass"] == s) & (canonical_annot["QBin"] == q)]
        rate.loc[s, q] = np.nan if len(sub) == 0 else sub["Supported"].mean()
mat = rate[QBINS].values.astype(float)
im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(QBINS)))
ax.set_xticklabels(QBINS)
ax.set_yticks(range(len(SIZE_CLASSES)))
ax.set_yticklabels(SIZE_CLASSES)
ax.set_title("Experimental support rate of canonical G4 intervals")
for i in range(len(SIZE_CLASSES)):
    for j in range(len(QBINS)):
        val = mat[i, j]
        txt = "NA" if np.isnan(val) else f"{val:.2f}"
        ax.text(j, i, txt, ha="center", va="center", color="white" if not np.isnan(val) and val > 0.55 else "black", fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Supported fraction")

# C: coordinate delta histogram
ax = axes[1, 0]
matched_only = canonical_annot.loc[canonical_annot["score_matched"]].copy()
for size_class in SIZE_CLASSES:
    sub = matched_only.loc[matched_only["SizeClass"] == size_class, "sum_delta"]
    ax.hist(sub, bins=np.arange(-0.5, max(4, sub.max()) + 1.5, 1), alpha=0.55, label=size_class, color=PALETTE_SIZE[size_class])
ax.set_title("Canonical→score matching precision")
ax.set_xlabel("|Δstart| + |Δend| (bp)")
ax.set_ylabel("Count")
ax.legend(frameon=False)

# D: score distribution by quartile
ax = axes[1, 1]
score_box = [matched_only.loc[matched_only["QBin"] == q, "Score"].dropna().values for q in ["Q1", "Q2", "Q3", "Q4"]]
ax.boxplot(score_box, patch_artist=True, labels=["Q1", "Q2", "Q3", "Q4"], showfliers=False)
for patch, q in zip(ax.artists if hasattr(ax, 'artists') else [], ["Q1", "Q2", "Q3", "Q4"]):
    patch.set_facecolor(PALETTE_Q[q])
ax.set_title("pqsfinder score distribution after quartile assignment")
ax.set_ylabel("Score")

f1_png = FIGDIR / "g4_annotation_qc.png"
f1_pdf = FIGDIR / "g4_annotation_qc.pdf"
fig.savefig(f1_png, dpi=300, bbox_inches="tight")
fig.savefig(f1_pdf, bbox_inches="tight")
plt.close(fig)
logger.info(f"[write] {f1_png}")
logger.info(f"[write] {f1_pdf}")

# -----------------------------
# Figure 2: window-level feature distributions
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
metrics = [
    ("g4exp__hits", "Experimental G4 hits per window"),
    ("g4exp__covfrac", "Experimental G4 coverage fraction"),
    ("g4seq_any_hits", "Canonical sequence-derived G4 hits per window"),
    ("g4seq_any_covfrac", "Canonical sequence-derived G4 union coverage fraction"),
]
for ax, (metric, title) in zip(axes.flat, metrics):
    positions = np.arange(1, 4)
    vals = [plot_df.loc[plot_df["WindowSet"] == ws, metric].values for ws in ["prdm9", "datasetA_matched", "datasetB_global"]]
    parts = ax.violinplot(vals, positions=positions, showmeans=True, showextrema=False, widths=0.85)
    for body, ws in zip(parts["bodies"], ["prdm9", "datasetA_matched", "datasetB_global"]):
        body.set_facecolor(PALETTE_WINDOWSET[ws])
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels(["PRDM9", "datasetA", "datasetB"])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Window-level G4 feature validation", fontsize=16)
f2_png = FIGDIR / "g4_window_feature_validation.png"
f2_pdf = FIGDIR / "g4_window_feature_validation.pdf"
fig.savefig(f2_png, dpi=300, bbox_inches="tight")
fig.savefig(f2_pdf, bbox_inches="tight")
plt.close(fig)
logger.info(f"[write] {f2_png}")
logger.info(f"[write] {f2_pdf}")

# -----------------------------
# Figure 3: strongest mean shifts on collapsed features
# -----------------------------
plot_ratio = ratio_df.copy()
plot_ratio = plot_ratio.sort_values("Log2Ratio_PRDM9_over_datasetA", ascending=False)
plot_ratio = pd.concat([plot_ratio.head(10), plot_ratio.tail(10)], ignore_index=True)
plot_ratio["Feature"] = pd.Categorical(plot_ratio["Feature"], categories=plot_ratio["Feature"].tolist()[::-1], ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.35 * len(plot_ratio))), constrained_layout=True)
for ax, col, color, title in [
    (axes[0], "Log2Ratio_PRDM9_over_datasetA", PALETTE_WINDOWSET["datasetA_matched"], "PRDM9 vs datasetA matched"),
    (axes[1], "Log2Ratio_PRDM9_over_datasetB", PALETTE_WINDOWSET["datasetB_global"], "PRDM9 vs datasetB global"),
]:
    ax.barh(plot_ratio["Feature"], plot_ratio[col], color=color, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("log2 mean ratio")
    ax.grid(axis="x", alpha=0.25)
fig.suptitle("Collapsed G4 features with strongest PRDM9/background mean shifts", fontsize=15)
f3_png = FIGDIR / "g4_feature_mean_shift_summary.png"
f3_pdf = FIGDIR / "g4_feature_mean_shift_summary.pdf"
fig.savefig(f3_png, dpi=300, bbox_inches="tight")
fig.savefig(f3_pdf, bbox_inches="tight")
plt.close(fig)
logger.info(f"[write] {f3_png}")
logger.info(f"[write] {f3_pdf}")

# summary table for the window-level validation metrics
val_rows = []
for ws, sub in plot_df.groupby("WindowSet", sort=False):
    for metric in ["g4exp__hits", "g4exp__covfrac", "g4seq_any_hits", "g4seq_any_covfrac"]:
        stats = summarize_numeric(sub[metric])
        stats.update({"WindowSet": ws, "Metric": metric})
        val_rows.append(stats)
val_table = pd.DataFrame(val_rows)
save_table(val_table, TABDIR / "g4_window_validation_metrics.tsv")

logger.info("All QC figures and validation tables written.")

# %% [markdown] cell 11
#
# ## Notes for downstream merging
#
# The main downstream-ready files are:
#
# - `features/g4_features_detailed.parquet`
# - `features/g4_features_collapsed.parquet`
#
# Both contain the merge keys:
#
# - `WindowSet`
# - `Chromosome`
# - `Start`
# - `End`
# - `window_id`
#
# Use the **detailed** table when support status matters.
# Use the **collapsed** table when you want the old-style features without splitting by experimental support.
#
# Because coverage can overlap across different feature groups, only per-group `covfrac` values are stored. For QC, the notebook separately computes `g4seq_any_covfrac` as a validation-only aggregate using true union coverage over all canonical sequence-derived G4 intervals.
