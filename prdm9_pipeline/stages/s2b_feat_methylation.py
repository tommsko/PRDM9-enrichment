"""Stage s2b_feat_methylation: converted from notebooks/2b_feat_methylation.ipynb by tools/convert_notebooks.py.

Run through the workflow (it sets the working directory and environment); see that converter for the
complete list of differences from the notebook. Edit the notebook + converter, not this file.
"""
from prdm9_pipeline.nbcompat import display, mc_iters, n_boot, threads  # noqa: F401

# %% [markdown] cell 0
#
# # PRDM9 methylation feature generation — production notebook
#
# This notebook builds the **methylation feature table** for all three window sets:
#
# - `prdm9`
# - `datasetA_matched`
# - `datasetB_global`
#
# using the merged RRBS track:
#
# - `methylation/HEK_RRBS_DNAmet_R1+R2_hg38.sorted.bed`
#
# ## Feature definition used here
#
# We keep the feature family intentionally compact and publication-safe:
#
# - `methylation__Q1__covfrac`
# - `methylation__Q2__covfrac`
# - `methylation__Q3__covfrac`
# - `methylation__Q4__covfrac`
#
# where quartiles are **fixed methylation bins** on RRBS methylation percentage:
#
# - `Q1`: `[0, 25)` — low / unmethylated
# - `Q2`: `[25, 50)`
# - `Q3`: `[50, 75)`
# - `Q4`: `[75, 100]` — highly / fully methylated
#
# and `covfrac` is the **fraction of window bases overlapped by RRBS intervals in that methylation quartile**.
#
# A few important methodological choices:
#
# 1. **The feature is interval overlap based, not just site-count based.**  
#    RRBS calls are typically 1-bp CpG intervals, so this is usually numerically similar to site-density / window length, but the interval formulation is the correct one and keeps the implementation honest.
#
# 2. **Duplicate RRBS coordinates are collapsed before feature generation.**  
#    If the merged RRBS file contains repeated genomic coordinates, they are collapsed by coordinate using a **coverage-weighted methylation mean**.
#
# 3. **All three window sets are processed together with the exact same RRBS handling.**
#
# 4. **We keep a `methylation__covered__covfrac` and `methylation__no_coverage` column in the final table.**  
#    These are not part of the main biological signal family, but they are extremely useful later because all-zero quartile features can otherwise mean either “no RRBS support” or “true absence”.
#
# ## Outputs
#
# - combined parquet with all windows
# - one parquet per window set
# - QC / validation tables
# - publication-style validation figures
# - metadata JSON
# - full log file

# %% cell 1

# ------------------------------------------------------------------
# Imports and configuration
# ------------------------------------------------------------------
import gc
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
# display() is provided by prdm9_pipeline.nbcompat

console = Console()

N_CPUS = threads(64)
MIN_RRBS_COV = 1

SAMPLING_DIR = Path("data_prdm9_sampling_production")
PRDM9_BED = SAMPLING_DIR / "prdm9_filtered.bed"
DATASETA_BED = SAMPLING_DIR / "datasetA_matched.len_gc_entropy.bed"
DATASETB_BED = SAMPLING_DIR / "datasetB_global.length_only.bed"

RRBS_BED = Path("methylation/HEK_RRBS_DNAmet_R1+R2_hg38.sorted.bed")

OUTDIR = Path("data_prdm9_methylation_production")
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"
FEATDIR = OUTDIR / "features"
METADIR = OUTDIR / "metadata"
LOGDIR = OUTDIR / "logs"
for p in [OUTDIR, FIGDIR, TABDIR, FEATDIR, METADIR, LOGDIR]:
    p.mkdir(parents=True, exist_ok=True)

ALL_PARQUET = FEATDIR / "methylation_features.parquet"
DEBUG_PARQUET = FEATDIR / "methylation_features_with_qc.parquet"
WINDOWS_TABLE = TABDIR / "all_windows.tsv"
RRBS_QC_TABLE = TABDIR / "rrbs_site_quartile_summary.tsv"
WINDOWSET_SUMMARY_TABLE = TABDIR / "methylation_windowset_summary.tsv"
RATIO_TABLE = TABDIR / "methylation_ratio_summary.tsv"
METADATA_JSON = METADIR / "methylation_feature_generation_metadata.json"
LOG_FILE = LOGDIR / "prdm9_methylation_feature_generation.log"

Q_LABELS = ["Q1", "Q2", "Q3", "Q4"]
PALETTE_WINDOWSET = {
    "prdm9": "#C44E52",
    "datasetA_matched": "#4C72B0",
    "datasetB_global": "#55A868",
}
PALETTE_Q = {
    "Q1": "#B2BEB5",
    "Q2": "#7A9CC6",
    "Q3": "#4F77B6",
    "Q4": "#1F3E75",
}

logger = logging.getLogger("prdm9_methylation")
logger.setLevel(logging.INFO)
logger.handlers.clear()

rich_handler = RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)
rich_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(rich_handler)

file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

console.print(Panel.fit(
    f"[bold]PRDM9 methylation production feature notebook[/bold]\n"
    f"N_CPUS = {N_CPUS}\n"
    f"MIN_RRBS_COV = {MIN_RRBS_COV}\n"
    f"RRBS = {RRBS_BED}\n"
    f"Output directory = {OUTDIR}"
))

# %% [markdown] cell 2
#
# ## Load windows and RRBS track
#
# The attached exploratory notebook was useful, but for the publication version we simplify the target to the exact feature family you described:
#
# - only Q1–Q4 methylation bins
# - `covfrac` per bin
# - same logic for PRDM9, datasetA, and datasetB
#
# The main extra care here is in **duplicate RRBS site handling** and **coverage-aware collapsing** before quartile assignment.

# %% cell 3

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def read_bed3(path, window_set):
    df = pd.read_csv(path, sep="\t", header=None, names=["Chromosome", "Start", "End"], usecols=[0, 1, 2])
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    df = df[df["End"] > df["Start"]].copy()
    df["WindowSet"] = window_set
    return df

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
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "q01": float(s.quantile(0.01)),
        "q05": float(s.quantile(0.05)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }

def assign_quartile(meth):
    meth = np.asarray(meth, dtype=np.float32)
    out = np.empty(len(meth), dtype=object)
    out[meth < 25.0] = "Q1"
    out[(meth >= 25.0) & (meth < 50.0)] = "Q2"
    out[(meth >= 50.0) & (meth < 75.0)] = "Q3"
    out[meth >= 75.0] = "Q4"
    return out

def safe_divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros_like(a, dtype=np.float64)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out

def validate_non_overlapping_intervals(df, chrom_col="Chromosome", start_col="Start", end_col="End", label="intervals"):
    bad = []
    for chrom, sub in df.groupby(chrom_col, sort=False):
        s = sub[[start_col, end_col]].sort_values([start_col, end_col]).to_numpy(np.int64)
        if len(s) <= 1:
            continue
        if np.any(s[1:, 0] < s[:-1, 1]):
            bad.append(chrom)
    if bad:
        raise ValueError(f"Overlapping {label} detected on chromosomes: {bad[:10]}")
    logger.info(f"[check] no within-chromosome overlaps in {label}")

def overlap_cov_nonoverlap(win_starts, win_ends, starts, ends):
    """
    Fast overlap length between windows and a non-overlapping interval set.
    All coordinates are half-open [start, end).
    """
    win_starts = np.asarray(win_starts, dtype=np.int64)
    win_ends = np.asarray(win_ends, dtype=np.int64)

    if len(starts) == 0:
        return np.zeros(len(win_starts), dtype=np.int64)

    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)

    lens = ends - starts
    prefix = np.concatenate([[0], np.cumsum(lens, dtype=np.int64)])

    i = np.searchsorted(ends, win_starts, side="right")
    j = np.searchsorted(starts, win_ends, side="left")

    out = np.zeros(len(win_starts), dtype=np.int64)
    mask = j > i
    if np.any(mask):
        total = prefix[j[mask]] - prefix[i[mask]]
        left_trim = np.maximum(0, win_starts[mask] - starts[i[mask]])
        right_trim = np.maximum(0, ends[j[mask] - 1] - win_ends[mask])
        out[mask] = total - left_trim - right_trim
    return out

def process_chromosome(task):
    chrom, win_sub, rrbs_sub = task

    win_sub = win_sub.sort_values(["Start", "End"]).reset_index(drop=True).copy()
    win_starts = win_sub["Start"].to_numpy(np.int64)
    win_ends = win_sub["End"].to_numpy(np.int64)
    win_lens = (win_ends - win_starts).astype(np.int64)

    rrbs_sub = rrbs_sub.sort_values(["Start", "End"]).reset_index(drop=True).copy()
    rrbs_starts = rrbs_sub["Start"].to_numpy(np.int64)
    rrbs_ends = rrbs_sub["End"].to_numpy(np.int64)
    rrbs_q = rrbs_sub["Quartile"].to_numpy(object)

    covered_bp = overlap_cov_nonoverlap(win_starts, win_ends, rrbs_starts, rrbs_ends)

    out = pd.DataFrame({
        "window_id": win_sub["window_id"].to_numpy(object),
        "Chromosome": chrom,
        "Start": win_starts,
        "End": win_ends,
        "window_len": win_lens,
        "methylation__covered_bp": covered_bp,
    })

    for q in Q_LABELS:
        mask = rrbs_q == q
        q_bp = overlap_cov_nonoverlap(win_starts, win_ends, rrbs_starts[mask], rrbs_ends[mask])
        out[f"methylation__{q}__bp"] = q_bp

    return out

def plot_ecdf(ax, values, label, color):
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0:
        return
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    ax.plot(x, y, label=label, color=color, linewidth=2)

def write_per_windowset(df, stem):
    for window_set, sub in df.groupby("WindowSet", sort=False):
        out = FEATDIR / f"{window_set}.{stem}.parquet"
        sub.to_parquet(out, index=False)
        logger.info(f"[write] {out} ({len(sub):,} rows)")

# ------------------------------------------------------------------
# Load windows
# ------------------------------------------------------------------
logger.info("[load] loading window sets")

prdm9 = read_bed3(PRDM9_BED, "prdm9")
datasetA = read_bed3(DATASETA_BED, "datasetA_matched")
datasetB = read_bed3(DATASETB_BED, "datasetB_global")

windows = pd.concat([prdm9, datasetA, datasetB], ignore_index=True)
windows = add_global_window_id(windows)
windows["window_len"] = (windows["End"] - windows["Start"]).astype(np.int64)

if windows["window_id"].duplicated().any():
    dup = int(windows["window_id"].duplicated().sum())
    raise ValueError(f"Duplicated window_id values detected: {dup}")

windows = windows.sort_values(["WindowSet", "Chromosome", "Start", "End"]).reset_index(drop=True)
windows.to_csv(WINDOWS_TABLE, sep="\t", index=False)

logger.info(f"[load] total windows: {len(windows):,}")
for ws, sub in windows.groupby("WindowSet", sort=False):
    logger.info(f"        {ws}: {len(sub):,}")

display(windows.head())

# ------------------------------------------------------------------
# Load RRBS
# ------------------------------------------------------------------
logger.info("[load] loading RRBS")
rrbs = pd.read_csv(
    RRBS_BED,
    sep="\t",
    header=None,
    comment="#",
    names=[
        "Chromosome", "Start", "End", "Name", "Score", "Strand",
        "ThickStart", "ThickEnd", "RGB", "Coverage", "MethPct"
    ],
    usecols=[0, 1, 2, 9, 10],
)

rrbs["Chromosome"] = rrbs["Chromosome"].astype(str)
rrbs["Start"] = pd.to_numeric(rrbs["Start"], errors="coerce").astype("Int64")
rrbs["End"] = pd.to_numeric(rrbs["End"], errors="coerce").astype("Int64")
rrbs["Coverage"] = pd.to_numeric(rrbs["Coverage"], errors="coerce")
rrbs["MethPct"] = pd.to_numeric(rrbs["MethPct"], errors="coerce")

n_raw_rrbs = len(rrbs)

rrbs = rrbs.dropna(subset=["Chromosome", "Start", "End", "Coverage", "MethPct"]).copy()
rrbs["Start"] = rrbs["Start"].astype(np.int64)
rrbs["End"] = rrbs["End"].astype(np.int64)
rrbs["Coverage"] = rrbs["Coverage"].astype(np.int64)
rrbs["MethPct"] = rrbs["MethPct"].astype(np.float32)

rrbs = rrbs[(rrbs["End"] > rrbs["Start"])].copy()
rrbs = rrbs[(rrbs["Coverage"] >= MIN_RRBS_COV)].copy()
rrbs = rrbs[(rrbs["MethPct"] >= 0.0) & (rrbs["MethPct"] <= 100.0)].copy()

logger.info(f"[load] RRBS raw rows: {n_raw_rrbs:,}")
logger.info(f"[load] RRBS after basic filters: {len(rrbs):,}")
logger.info(f"[load] RRBS methylation summary: {summarize_numeric(rrbs['MethPct'])}")
logger.info(f"[load] RRBS coverage summary: {summarize_numeric(rrbs['Coverage'])}")

# collapse duplicate genomic coordinates, if any
dup_mask = rrbs.duplicated(["Chromosome", "Start", "End"], keep=False)
n_dup_rows = int(dup_mask.sum())

if n_dup_rows > 0:
    logger.info(f"[collapse] collapsing duplicate RRBS coordinates: {n_dup_rows:,} duplicate rows")
    rrbs["MethTimesCov"] = rrbs["MethPct"] * rrbs["Coverage"]
    rrbs_collapsed = (
        rrbs.groupby(["Chromosome", "Start", "End"], sort=False, as_index=False)
            .agg(
                Coverage=("Coverage", "sum"),
                MethTimesCov=("MethTimesCov", "sum"),
                n_rows=("MethPct", "size"),
                MethPct_mean=("MethPct", "mean"),
            )
    )
    rrbs_collapsed["MethPct"] = np.where(
        rrbs_collapsed["Coverage"] > 0,
        rrbs_collapsed["MethTimesCov"] / rrbs_collapsed["Coverage"],
        rrbs_collapsed["MethPct_mean"],
    ).astype(np.float32)
    rrbs = rrbs_collapsed[["Chromosome", "Start", "End", "Coverage", "MethPct"]].copy()
else:
    logger.info("[collapse] no duplicate RRBS coordinates detected")

rrbs["Quartile"] = assign_quartile(rrbs["MethPct"])

validate_non_overlapping_intervals(rrbs, label="collapsed RRBS intervals")

rrbs_qc = (
    rrbs.groupby("Quartile", sort=False)
        .agg(
            n_sites=("Quartile", "size"),
            bp_covered=("Start", "size"),
            mean_meth=("MethPct", "mean"),
            median_meth=("MethPct", "median"),
            mean_cov=("Coverage", "mean"),
            median_cov=("Coverage", "median"),
        )
        .reset_index()
)
rrbs_qc["site_fraction"] = rrbs_qc["n_sites"] / rrbs_qc["n_sites"].sum()
rrbs_qc.to_csv(RRBS_QC_TABLE, sep="\t", index=False)

display(rrbs_qc)

# %% [markdown] cell 4
#
# ## Parallel overlap engine
#
# We process one chromosome per worker and compute interval overlap coverage against:
#
# - all RRBS intervals combined
# - RRBS intervals in `Q1`
# - RRBS intervals in `Q2`
# - RRBS intervals in `Q3`
# - RRBS intervals in `Q4`
#
# The overlap engine assumes the RRBS intervals are non-overlapping after duplicate-coordinate collapse, which we already validated above.

# %% cell 5

# ------------------------------------------------------------------
# Parallel feature generation
# ------------------------------------------------------------------
logger.info("[feature-gen] preparing chromosome tasks")

chroms = sorted(set(windows["Chromosome"]).intersection(set(rrbs["Chromosome"])))
if len(chroms) == 0:
    raise RuntimeError("No shared chromosomes between windows and RRBS track")

tasks = []
for chrom in chroms:
    win_sub = windows[windows["Chromosome"] == chrom][["window_id", "Chromosome", "Start", "End", "window_len"]].copy()
    rrbs_sub = rrbs[rrbs["Chromosome"] == chrom][["Chromosome", "Start", "End", "Quartile"]].copy()
    tasks.append((chrom, win_sub, rrbs_sub))

logger.info(f"[feature-gen] chromosomes to process: {len(tasks):,}")
logger.info(f"[feature-gen] workers requested: {N_CPUS}")

parts = []
with ProcessPoolExecutor(max_workers=min(N_CPUS, len(tasks))) as ex:
    fut_to_chrom = {ex.submit(process_chromosome, task): task[0] for task in tasks}
    for fut in tqdm(as_completed(fut_to_chrom), total=len(fut_to_chrom), desc="chromosomes", mininterval=0.5):
        chrom = fut_to_chrom[fut]
        part = fut.result()
        logger.info(f"[feature-gen] finished {chrom}: {len(part):,} windows")
        parts.append(part)

feat_bp = pd.concat(parts, ignore_index=True)
del parts
gc.collect()

logger.info(f"[feature-gen] raw feature rows: {len(feat_bp):,}")

# Merge back to full windows table, fill zeros for chromosomes with no RRBS support
features = windows.merge(
    feat_bp.drop(columns=["Chromosome", "Start", "End"]),
    on=["window_id", "window_len"],
    how="left",
)

bp_cols = ["methylation__covered_bp"] + [f"methylation__{q}__bp" for q in Q_LABELS]
for c in bp_cols:
    features[c] = features[c].fillna(0).astype(np.int64)

for q in Q_LABELS:
    bp_col = f"methylation__{q}__bp"
    cov_col = f"methylation__{q}__covfrac"
    features[cov_col] = safe_divide(features[bp_col].to_numpy(np.int64), features["window_len"].to_numpy(np.int64)).astype(np.float32)

features["methylation__covered__covfrac"] = safe_divide(
    features["methylation__covered_bp"].to_numpy(np.int64),
    features["window_len"].to_numpy(np.int64)
).astype(np.float32)

features["methylation__no_coverage"] = (features["methylation__covered_bp"].to_numpy(np.int64) == 0).astype(np.int8)

# exact partition check
features["methylation__quartile_sum_bp"] = features[[f"methylation__{q}__bp" for q in Q_LABELS]].sum(axis=1).astype(np.int64)
features["methylation__quartile_sum__covfrac"] = features[[f"methylation__{q}__covfrac" for q in Q_LABELS]].sum(axis=1).astype(np.float32)

if not np.array_equal(
    features["methylation__quartile_sum_bp"].to_numpy(np.int64),
    features["methylation__covered_bp"].to_numpy(np.int64)
):
    diff = int(np.abs(
        features["methylation__quartile_sum_bp"].to_numpy(np.int64)
        - features["methylation__covered_bp"].to_numpy(np.int64)
    ).sum())
    raise AssertionError(f"Quartile bp sums do not match covered bp. Total absolute discrepancy = {diff}")

# bounds / sanity checks
for c in [f"methylation__{q}__covfrac" for q in Q_LABELS] + ["methylation__covered__covfrac", "methylation__quartile_sum__covfrac"]:
    vals = features[c].to_numpy(np.float64)
    if np.any(vals < -1e-12) or np.any(vals > 1.0 + 1e-12):
        raise AssertionError(f"Covfrac bounds violated in {c}")

if not np.allclose(
    features["methylation__quartile_sum__covfrac"].to_numpy(np.float64),
    features["methylation__covered__covfrac"].to_numpy(np.float64),
    atol=1e-8,
):
    raise AssertionError("Quartile covfrac sum does not match covered covfrac")

if features["window_id"].duplicated().any():
    raise AssertionError("Duplicated window_id rows in final feature table")

# within-covered composition for validation only
for q in Q_LABELS:
    features[f"methylation__{q}__within_covered"] = np.where(
        features["methylation__covered__covfrac"].to_numpy(np.float32) > 0,
        features[f"methylation__{q}__covfrac"].to_numpy(np.float32) / features["methylation__covered__covfrac"].to_numpy(np.float32),
        np.nan,
    ).astype(np.float32)

logger.info("[feature-gen] feature generation checks passed")

# order columns
main_cols = [
    "WindowSet", "Chromosome", "Start", "End", "window_len", "window_id",
    "methylation__Q1__covfrac",
    "methylation__Q2__covfrac",
    "methylation__Q3__covfrac",
    "methylation__Q4__covfrac",
    "methylation__covered__covfrac",
    "methylation__no_coverage",
]
aux_cols = [
    "methylation__Q1__bp", "methylation__Q2__bp", "methylation__Q3__bp", "methylation__Q4__bp",
    "methylation__covered_bp",
    "methylation__quartile_sum_bp", "methylation__quartile_sum__covfrac",
    "methylation__Q1__within_covered", "methylation__Q2__within_covered",
    "methylation__Q3__within_covered", "methylation__Q4__within_covered",
]
features = features[main_cols + aux_cols].sort_values(["WindowSet", "Chromosome", "Start", "End"]).reset_index(drop=True)

features_main = features[main_cols].copy()

features_main.to_parquet(ALL_PARQUET, index=False)
features.to_parquet(DEBUG_PARQUET, index=False)
write_per_windowset(features_main, stem="methylation_features")

logger.info(f"[write] {ALL_PARQUET} ({len(features_main):,} rows)")
logger.info(f"[write] {DEBUG_PARQUET} ({len(features):,} rows)")

display(features_main.head())

# %% [markdown] cell 6
#
# ## QC summaries and publication-style validation figures
#
# The figures here are focused on the checks that actually matter for this feature family:
#
# 1. the global RRBS methylation landscape and quartile partition
# 2. how much RRBS-covered sequence each window set has
# 3. how the methylation composition of covered sequence differs across PRDM9 and the two backgrounds
# 4. direct PRDM9/background mean-ratio summaries

# %% cell 7

# ------------------------------------------------------------------
# QC tables
# ------------------------------------------------------------------
summary_rows = []
for ws, sub in features.groupby("WindowSet", sort=False):
    row = {"WindowSet": ws, "n_windows": int(len(sub))}
    for col in [
        "methylation__Q1__covfrac",
        "methylation__Q2__covfrac",
        "methylation__Q3__covfrac",
        "methylation__Q4__covfrac",
        "methylation__covered__covfrac",
        "methylation__no_coverage",
    ]:
        row[f"{col}__mean"] = float(sub[col].mean())
        row[f"{col}__median"] = float(sub[col].median())
        row[f"{col}__q95"] = float(sub[col].quantile(0.95))
        row[f"{col}__q99"] = float(sub[col].quantile(0.99))
    covered_mask = sub["methylation__covered__covfrac"] > 0
    for q in Q_LABELS:
        x = sub.loc[covered_mask, f"methylation__{q}__within_covered"]
        row[f"methylation__{q}__within_covered__mean"] = float(x.mean()) if len(x) else np.nan
    summary_rows.append(row)

windowset_summary = pd.DataFrame(summary_rows)
windowset_summary.to_csv(WINDOWSET_SUMMARY_TABLE, sep="\t", index=False)
display(windowset_summary)

eps = 1e-9
means = (
    features.groupby("WindowSet", sort=False)[
        [f"methylation__{q}__covfrac" for q in Q_LABELS] + ["methylation__covered__covfrac"]
    ]
    .mean()
)

ratio_rows = []
for feature in means.columns:
    ratio_rows.append({
        "Feature": feature,
        "PRDM9_mean": float(means.loc["prdm9", feature]),
        "datasetA_mean": float(means.loc["datasetA_matched", feature]),
        "datasetB_mean": float(means.loc["datasetB_global", feature]),
        "Log2Ratio_PRDM9_over_datasetA": float(np.log2((means.loc["prdm9", feature] + eps) / (means.loc["datasetA_matched", feature] + eps))),
        "Log2Ratio_PRDM9_over_datasetB": float(np.log2((means.loc["prdm9", feature] + eps) / (means.loc["datasetB_global", feature] + eps))),
    })
ratio_df = pd.DataFrame(ratio_rows)
ratio_df.to_csv(RATIO_TABLE, sep="\t", index=False)
display(ratio_df)

# metadata json
metadata = {
    "n_windows_total": int(len(features)),
    "n_windows_by_set": {k: int(v) for k, v in windows["WindowSet"].value_counts().to_dict().items()},
    "n_rrbs_raw_rows": int(n_raw_rrbs),
    "n_rrbs_after_filtering": int(len(rrbs)),
    "n_rrbs_duplicate_rows_collapsed": int(n_dup_rows),
    "rrbs_methylation_summary": summarize_numeric(rrbs["MethPct"]),
    "rrbs_coverage_summary": summarize_numeric(rrbs["Coverage"]),
    "rrbs_quartile_counts": {r["Quartile"]: int(r["n_sites"]) for _, r in rrbs_qc.iterrows()},
    "output_files": {
        "features_main": str(ALL_PARQUET),
        "features_debug": str(DEBUG_PARQUET),
    },
    "feature_columns_main": [
        "methylation__Q1__covfrac",
        "methylation__Q2__covfrac",
        "methylation__Q3__covfrac",
        "methylation__Q4__covfrac",
        "methylation__covered__covfrac",
        "methylation__no_coverage",
    ],
}
with open(METADATA_JSON, "w") as f:
    json.dump(metadata, f, indent=2)

logger.info(f"[write] {WINDOWSET_SUMMARY_TABLE}")
logger.info(f"[write] {RATIO_TABLE}")
logger.info(f"[write] {METADATA_JSON}")

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
# Figure 1: global RRBS QC
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

ax = axes[0]
bins = np.arange(0, 101, 1)
ax.hist(rrbs["MethPct"], bins=bins, edgecolor="none", alpha=0.9, color="#6E6E6E")
for x in [25, 50, 75]:
    ax.axvline(x, color="black", linestyle="--", linewidth=1)
for q, (a, b) in {"Q1": (0, 25), "Q2": (25, 50), "Q3": (50, 75), "Q4": (75, 100)}.items():
    ax.axvspan(a, b, color=PALETTE_Q[q], alpha=0.12)
ax.set_title("RRBS methylation distribution")
ax.set_xlabel("Methylation percentage")
ax.set_ylabel("RRBS sites")
ax.grid(alpha=0.2)

ax = axes[1]
rrbs_plot = rrbs_qc.set_index("Quartile").loc[Q_LABELS].reset_index()
ax.bar(rrbs_plot["Quartile"], rrbs_plot["site_fraction"], color=[PALETTE_Q[q] for q in rrbs_plot["Quartile"]], edgecolor="black", linewidth=0.5)
for i, row in rrbs_plot.iterrows():
    ax.text(i, row["site_fraction"] + 0.01, f"{row['site_fraction']*100:.1f}%", ha="center", va="bottom", fontsize=10)
ax.set_title("RRBS site fraction by methylation quartile")
ax.set_xlabel("Quartile")
ax.set_ylabel("Fraction of RRBS sites")
ax.set_ylim(0, max(rrbs_plot["site_fraction"]) * 1.15)
ax.grid(axis="y", alpha=0.2)

fig.suptitle("Global RRBS methylation landscape", fontsize=15)
f1_png = FIGDIR / "methylation_rrbs_global_qc.png"
f1_pdf = FIGDIR / "methylation_rrbs_global_qc.pdf"
fig.savefig(f1_png, dpi=300, bbox_inches="tight")
fig.savefig(f1_pdf, bbox_inches="tight")
plt.close(fig)
logger.info(f"[write] {f1_png}")
logger.info(f"[write] {f1_pdf}")

# Figure 2: ECDF distributions across window sets
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
axes = axes.ravel()

ecdf_features = [
    "methylation__covered__covfrac",
    "methylation__Q1__covfrac",
    "methylation__Q2__covfrac",
    "methylation__Q3__covfrac",
    "methylation__Q4__covfrac",
]
titles = {
    "methylation__covered__covfrac": "RRBS-covered fraction",
    "methylation__Q1__covfrac": "Q1 covfrac",
    "methylation__Q2__covfrac": "Q2 covfrac",
    "methylation__Q3__covfrac": "Q3 covfrac",
    "methylation__Q4__covfrac": "Q4 covfrac",
}

for ax, feat in zip(axes[:5], ecdf_features):
    for ws in ["prdm9", "datasetA_matched", "datasetB_global"]:
        vals = features.loc[features["WindowSet"] == ws, feat].to_numpy(np.float32)
        plot_ecdf(ax, vals, ws, PALETTE_WINDOWSET[ws])
    ax.set_title(titles[feat])
    ax.set_xlabel("covfrac")
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.2)

axes[0].legend(frameon=False, loc="lower right")

# final panel: no coverage fraction
ax = axes[5]
no_cov = (
    features.groupby("WindowSet", sort=False)["methylation__no_coverage"]
    .mean()
    .reindex(["prdm9", "datasetA_matched", "datasetB_global"])
)
ax.bar(
    no_cov.index,
    no_cov.values,
    color=[PALETTE_WINDOWSET[k] for k in no_cov.index],
    edgecolor="black",
    linewidth=0.5,
)
for i, v in enumerate(no_cov.values):
    ax.text(i, v + 0.01, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=10)
ax.set_title("Windows with no RRBS support")
ax.set_ylabel("Fraction of windows")
ax.set_ylim(0, max(no_cov.values) * 1.15 if len(no_cov) else 1.0)
ax.grid(axis="y", alpha=0.2)

fig.suptitle("Window-level methylation feature distributions", fontsize=15)
f2_png = FIGDIR / "methylation_window_feature_distributions.png"
f2_pdf = FIGDIR / "methylation_window_feature_distributions.pdf"
fig.savefig(f2_png, dpi=300, bbox_inches="tight")
fig.savefig(f2_pdf, bbox_inches="tight")
plt.close(fig)
logger.info(f"[write] {f2_png}")
logger.info(f"[write] {f2_pdf}")

# Figure 3: composition + mean ratio summary
composition = []
for ws, sub in features.groupby("WindowSet", sort=False):
    covered = sub["methylation__covered__covfrac"].to_numpy(np.float32)
    row = {"WindowSet": ws}
    for q in Q_LABELS:
        vals = sub[f"methylation__{q}__within_covered"].dropna()
        row[q] = float(vals.mean()) if len(vals) else np.nan
    composition.append(row)
composition_df = pd.DataFrame(composition).set_index("WindowSet").loc[["prdm9", "datasetA_matched", "datasetB_global"]]

fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

# 3A stacked composition
ax = axes[0]
bottom = np.zeros(len(composition_df), dtype=np.float64)
x = np.arange(len(composition_df))
for q in Q_LABELS:
    vals = composition_df[q].to_numpy(np.float64)
    ax.bar(x, vals, bottom=bottom, color=PALETTE_Q[q], edgecolor="black", linewidth=0.4, label=q)
    bottom += np.nan_to_num(vals, nan=0.0)
ax.set_xticks(x)
ax.set_xticklabels(composition_df.index, rotation=15)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Mean fraction within RRBS-covered bases")
ax.set_title("Methylation composition of covered sequence")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

# 3B mean covered fraction
ax = axes[1]
covered_mean = means.loc[["prdm9", "datasetA_matched", "datasetB_global"], "methylation__covered__covfrac"]
ax.bar(
    covered_mean.index,
    covered_mean.values,
    color=[PALETTE_WINDOWSET[k] for k in covered_mean.index],
    edgecolor="black",
    linewidth=0.5,
)
for i, v in enumerate(covered_mean.values):
    ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_title("Mean RRBS-covered fraction")
ax.set_ylabel("Mean covfrac")
ax.grid(axis="y", alpha=0.2)
ax.tick_params(axis="x", rotation=15)

# 3C log2 mean-ratio heatmap
ax = axes[2]
heat = ratio_df.set_index("Feature")[["Log2Ratio_PRDM9_over_datasetA", "Log2Ratio_PRDM9_over_datasetB"]].loc[
    [
        "methylation__Q1__covfrac",
        "methylation__Q2__covfrac",
        "methylation__Q3__covfrac",
        "methylation__Q4__covfrac",
        "methylation__covered__covfrac",
    ]
]
im = ax.imshow(heat.to_numpy(np.float64), aspect="auto", cmap="coolwarm")
ax.set_xticks(np.arange(heat.shape[1]))
ax.set_xticklabels(["PRDM9 / datasetA", "PRDM9 / datasetB"], rotation=15)
ax.set_yticks(np.arange(heat.shape[0]))
ax.set_yticklabels(["Q1", "Q2", "Q3", "Q4", "covered"])
ax.set_title("log2 mean ratio summary")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Methylation feature summary across PRDM9 and background window sets", fontsize=15)
f3_png = FIGDIR / "methylation_feature_summary.png"
f3_pdf = FIGDIR / "methylation_feature_summary.pdf"
fig.savefig(f3_png, dpi=300, bbox_inches="tight")
fig.savefig(f3_pdf, bbox_inches="tight")
plt.close(fig)
logger.info(f"[write] {f3_png}")
logger.info(f"[write] {f3_pdf}")

# compact terminal summary
tbl = Table(title="Window-set methylation summary")
tbl.add_column("WindowSet")
tbl.add_column("covered mean", justify="right")
tbl.add_column("Q1 mean", justify="right")
tbl.add_column("Q4 mean", justify="right")
tbl.add_column("no coverage", justify="right")

for ws in ["prdm9", "datasetA_matched", "datasetB_global"]:
    sub = windowset_summary[windowset_summary["WindowSet"] == ws].iloc[0]
    tbl.add_row(
        ws,
        f"{sub['methylation__covered__covfrac__mean']:.4f}",
        f"{sub['methylation__Q1__covfrac__mean']:.4f}",
        f"{sub['methylation__Q4__covfrac__mean']:.4f}",
        f"{sub['methylation__no_coverage__mean']:.2%}",
    )
console.print(tbl)

logger.info("[done] methylation production notebook finished successfully")
