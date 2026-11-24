"""Stage s2g_combine: converted from notebooks/2g_combine.ipynb by tools/convert_notebooks.py.

Run through the workflow (it sets the working directory and environment); see that converter for the
complete list of differences from the notebook. Edit the notebook + converter, not this file.
"""
from __future__ import annotations
from prdm9_pipeline.nbcompat import display, mc_iters, n_boot, threads  # noqa: F401

# %% [markdown] cell 0
#
# # PRDM9 project — production feature combination and analysis-ready dataset assembly
#
# This notebook combines all upstream feature families for the three window sets:
#
# - `prdm9`
# - `datasetA_matched`
# - `datasetB_global`
#
# It then creates publication-safe, analysis-ready wide tables for both comparisons:
#
# - `prdm9` vs `datasetA_matched`
# - `prdm9` vs `datasetB_global`
#
# Two G4 variants are handled explicitly:
#
# - **detailed**: retains `supported/unsupported`
# - **collapsed**: sums over support state
#
# The key production safeguards here are:
#
# 1. **Do not trust upstream `window_id` blindly.** Some upstream notebooks use string IDs, others integer IDs.  
#    We therefore standardize all joins on a canonical coordinate key:
#
#    `WindowSet::Chromosome:Start-End`
#
# 2. **Do not silently zero-fill merge failures.**  
#    Every feature block must cover the full window universe exactly once. Missing rows are treated as errors.
#
# 3. **Create separate comparison datasets.**  
#    `datasetA_matched` and `datasetB_global` are exported separately so ML and enrichment analyses can stay clean and interpretable.
#
# 4. **Keep a complete audit trail.**  
#    We write block manifests, feature catalogs, effect summaries, metadata JSON, and validation figures.

# %% cell 1


import os
import json
import math
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
# display() is provided by prdm9_pipeline.nbcompat

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
N_JOBS = threads(64)
RANDOM_STATE = 42
TOP_N_HEATMAP = 40
TOP_N_PER_BLOCK = 8
EPS = 1e-9

SAMPLING_DIR = Path("data_prdm9_sampling_production")
OUTDIR = Path("data_prdm9_combined_production")
LOGDIR = OUTDIR / "logs"
METADIR = OUTDIR / "metadata"
TABDIR = OUTDIR / "tables"

for d in [OUTDIR, LOGDIR, METADIR, TABDIR]:
    d.mkdir(parents=True, exist_ok=True)

BLOCK_PATHS = {
    "g4_detailed": Path("data_prdm9_g4_production/features/g4_features_detailed.parquet"),
    "g4_collapsed": Path("data_prdm9_g4_production/features/g4_features_collapsed.parquet"),
    "methylation": Path("data_prdm9_methylation_production/features/methylation_features.parquet"),
    "motifs": Path("data_prdm9_motifs_production/features/motif_features.parquet"),
    "accessibility": Path("data_prdm9_accessibility_production/features/accessibility_features.parquet"),
    "histones": Path("data_prdm9_histones_production/features/histone_features.parquet"),
    "recombination": Path("data_prdm9_recombination_production/features/recombination_features.parquet"),
}

PRDM9_BED = SAMPLING_DIR / "prdm9_filtered.bed"
DATASETA_BED = SAMPLING_DIR / "datasetA_matched.len_gc_entropy.bed"
DATASETB_BED = SAMPLING_DIR / "datasetB_global.length_only.bed"

console = Console()

logger = logging.getLogger("prdm9_combine")
logger.setLevel(logging.INFO)
logger.handlers = []

rich_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
rich_handler.setLevel(logging.INFO)

file_handler = RotatingFileHandler(LOGDIR / "combine_run.log", maxBytes=5_000_000, backupCount=3)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

logger.addHandler(rich_handler)
logger.addHandler(file_handler)

logger.info("Starting combined feature assembly notebook")
logger.info("Configured worker count: %d", N_JOBS)

for name, path in BLOCK_PATHS.items():
    logger.info("Expected block [%s]: %s", name, path)

# %% cell 2

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
ID_COLS = ["WindowSet", "Chromosome", "Start", "End"]
NON_FEATURE_COLS = {
    "WindowSet", "Chromosome", "Start", "End",
    "window_id", "window_uid", "window_index_global",
    "window_length", "window_length_bp", "window_len",
    "window_order_within_set", "label", "y",
    "comparison", "background_set", "group_label",
    "sample_weight_balanced",
}

BLOCK_ORDER = ["g4", "motifs", "methylation", "accessibility", "histones", "recombination", "other"]
WINDOWSET_ORDER = ["prdm9", "datasetA_matched", "datasetB_global"]

def read_bed3(path: Path, window_set: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["Chromosome", "Start", "End"])
    df["WindowSet"] = window_set
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    df["window_length_bp"] = (df["End"] - df["Start"]).astype(np.int64)
    return df[["WindowSet", "Chromosome", "Start", "End", "window_length_bp"]]

def make_window_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["WindowSet"].astype(str)
        + "::"
        + df["Chromosome"].astype(str)
        + ":"
        + df["Start"].astype(str)
        + "-"
        + df["End"].astype(str)
    )

def build_window_manifest() -> pd.DataFrame:
    prdm9 = read_bed3(PRDM9_BED, "prdm9")
    datasetA = read_bed3(DATASETA_BED, "datasetA_matched")
    datasetB = read_bed3(DATASETB_BED, "datasetB_global")

    parts = [prdm9, datasetA, datasetB]
    for df in parts:
        df["window_uid"] = make_window_uid(df)

    windows = pd.concat(parts, ignore_index=True)
    if windows["window_uid"].duplicated().any():
        dup = windows.loc[windows["window_uid"].duplicated(keep=False)].head()
        raise ValueError(f"Duplicate canonical window IDs detected:\n{dup}")

    windows["window_order_within_set"] = windows.groupby("WindowSet").cumcount().astype(np.int64)
    windows["window_index_global"] = np.arange(len(windows), dtype=np.int64)
    windows = windows[["window_uid", "window_index_global", "window_order_within_set", "WindowSet", "Chromosome", "Start", "End", "window_length_bp"]]
    return windows

def guess_block(feature_name: str) -> str:
    if feature_name.startswith("g4"):
        return "g4"
    if feature_name.startswith("motif__"):
        return "motifs"
    if feature_name.startswith("methylation__"):
        return "methylation"
    if feature_name.startswith("accessibility__"):
        return "accessibility"
    if feature_name.startswith("histone__"):
        return "histones"
    if feature_name.startswith("recombination__"):
        return "recombination"
    return "other"

def guess_metric(feature_name: str) -> str:
    parts = str(feature_name).split("__")
    return parts[-1] if len(parts) >= 2 else feature_name

def guess_family(feature_name: str) -> str:
    parts = str(feature_name).split("__")
    return "__".join(parts[:-1]) if len(parts) >= 2 else feature_name

def normalize_feature_block(path: Path, block_name: str, expected_windows: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature block: {path}")

    df = pd.read_parquet(path)
    logger.info("[%s] raw shape: %s", block_name, df.shape)

    missing_id = [c for c in ID_COLS if c not in df.columns]
    if missing_id:
        raise ValueError(f"[{block_name}] missing identifier columns: {missing_id}")

    df = df.copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    df["window_uid"] = make_window_uid(df)

    if df["window_uid"].duplicated().any():
        dup = df.loc[df["window_uid"].duplicated(keep=False), ID_COLS].head()
        raise ValueError(f"[{block_name}] duplicate windows detected:\n{dup}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if not feature_cols:
        raise ValueError(f"[{block_name}] no feature columns found")

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"[{block_name}] non-numeric feature columns detected: {non_numeric[:10]}")

    for c in feature_cols:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(np.uint8)
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        else:
            df[c] = pd.to_numeric(df[c], downcast="float")

    observed = set(df["window_uid"])
    expected = set(expected_windows["window_uid"])
    if observed != expected:
        only_expected = sorted(expected - observed)[:5]
        only_observed = sorted(observed - expected)[:5]
        raise ValueError(
            f"[{block_name}] window universe mismatch. "
            f"Missing from block: {only_expected}; unexpected in block: {only_observed}"
        )

    merged = expected_windows[["window_uid"] + ID_COLS].merge(
        df[["window_uid"] + feature_cols],
        on="window_uid",
        how="left",
        validate="one_to_one",
    )
    if merged[feature_cols].isna().any().any():
        bad = merged.loc[merged[feature_cols].isna().any(axis=1), ["window_uid"] + ID_COLS].head()
        raise ValueError(f"[{block_name}] merge introduced missing feature values:\n{bad}")

    finite_mask = np.isfinite(merged[feature_cols].to_numpy(dtype=np.float64))
    if not finite_mask.all():
        raise ValueError(f"[{block_name}] non-finite values detected in feature block")

    meta = {
        "block_name": block_name,
        "source_path": str(path),
        "n_rows": int(len(merged)),
        "n_features": int(len(feature_cols)),
    }
    logger.info("[%s] normalized rows=%d features=%d", block_name, len(merged), len(feature_cols))
    return merged, feature_cols, meta

def make_variant_outdirs(variant_name: str) -> dict[str, Path]:
    root = OUTDIR / variant_name
    out = {
        "root": root,
        "features": root / "features",
        "tables": root / "tables",
        "figures": root / "figures",
        "metadata": root / "metadata",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out

def build_comparison_dataset(all_df: pd.DataFrame, background_set: str, feature_cols: list[str]) -> pd.DataFrame:
    sub = all_df.loc[all_df["WindowSet"].isin(["prdm9", background_set])].copy()
    sub["y"] = (sub["WindowSet"] == "prdm9").astype(np.int8)
    sub["label"] = sub["y"]
    sub["comparison"] = f"prdm9_vs_{background_set}"
    sub["background_set"] = background_set
    sub["group_label"] = np.where(sub["WindowSet"] == "prdm9", "prdm9", "background")

    n_pos = int((sub["y"] == 1).sum())
    n_neg = int((sub["y"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Degenerate comparison dataset for background_set={background_set}")

    sub["sample_weight_balanced"] = 1.0
    sub.loc[sub["y"] == 0, "sample_weight_balanced"] = n_pos / n_neg

    ordered_cols = [
        "window_uid", "window_index_global", "window_order_within_set",
        "comparison", "background_set", "group_label", "WindowSet",
        "Chromosome", "Start", "End", "window_length_bp",
        "y", "label", "sample_weight_balanced",
    ] + feature_cols
    return sub[ordered_cols].reset_index(drop=True)

def compute_comparison_effects(df: pd.DataFrame, feature_cols: list[str], background_set: str) -> pd.DataFrame:
    pos = df.loc[df["WindowSet"] == "prdm9", feature_cols].to_numpy(dtype=np.float64)
    neg = df.loc[df["WindowSet"] == background_set, feature_cols].to_numpy(dtype=np.float64)

    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    pos_std = pos.std(axis=0, ddof=0)
    neg_std = neg.std(axis=0, ddof=0)
    pooled = np.sqrt((pos_std ** 2 + neg_std ** 2) / 2.0)
    cohens_d = np.divide(pos_mean - neg_mean, pooled, out=np.zeros_like(pos_mean), where=pooled > 0)

    pos_prev = (pos != 0).mean(axis=0)
    neg_prev = (neg != 0).mean(axis=0)

    out = pd.DataFrame({
        "feature": feature_cols,
        "comparison": f"prdm9_vs_{background_set}",
        "mean_prdm9": pos_mean,
        f"mean_{background_set}": neg_mean,
        "std_prdm9": pos_std,
        f"std_{background_set}": neg_std,
        "prevalence_prdm9": pos_prev,
        f"prevalence_{background_set}": neg_prev,
        "cohens_d": cohens_d,
        "abs_cohens_d": np.abs(cohens_d),
        "log2_mean_ratio": np.log2((pos_mean + EPS) / (neg_mean + EPS)),
        "log2_prevalence_ratio": np.log2((pos_prev + EPS) / (neg_prev + EPS)),
    })
    return out

def summarize_feature_chunk(all_df: pd.DataFrame, feature_chunk: list[str]) -> pd.DataFrame:
    out_rows = []
    feature_view = all_df[feature_chunk]
    for c in feature_chunk:
        arr = feature_view[c].to_numpy(dtype=np.float64)
        zero_frac = float((arr == 0).mean())
        out_rows.append({
            "feature": c,
            "block": guess_block(c),
            "family": guess_family(c),
            "metric": guess_metric(c),
            "dtype": str(feature_view[c].dtype),
            "mean_all": float(np.mean(arr)),
            "std_all": float(np.std(arr, ddof=0)),
            "min_all": float(np.min(arr)),
            "max_all": float(np.max(arr)),
            "zero_frac_all": zero_frac,
            "nonzero_frac_all": 1.0 - zero_frac,
        })
    return pd.DataFrame(out_rows)

def parallel_feature_catalog(all_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if not feature_cols:
        raise ValueError("No feature columns supplied to catalog builder")

    chunk_size = max(1, math.ceil(len(feature_cols) / N_JOBS))
    chunks = [feature_cols[i:i + chunk_size] for i in range(0, len(feature_cols), chunk_size)]

    parts = []
    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        futures = [ex.submit(summarize_feature_chunk, all_df, chunk) for chunk in chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Feature catalog", leave=False):
            parts.append(fut.result())

    catalog = pd.concat(parts, ignore_index=True)
    catalog["block"] = pd.Categorical(catalog["block"], categories=BLOCK_ORDER, ordered=True)
    catalog = catalog.sort_values(["block", "family", "metric", "feature"], kind="mergesort").reset_index(drop=True)
    return catalog

def make_overview_figure(all_df: pd.DataFrame, feature_catalog: pd.DataFrame, out_png: Path, title_suffix: str):
    block_counts = feature_catalog.groupby("block", observed=True)["feature"].count().reindex(BLOCK_ORDER).dropna()
    block_zero = feature_catalog.groupby("block", observed=True)["zero_frac_all"].mean().reindex(BLOCK_ORDER).dropna()
    set_counts = all_df["WindowSet"].value_counts().reindex(WINDOWSET_ORDER)

    fig = plt.figure(figsize=(13, 4.6), dpi=180)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(set_counts.index, set_counts.values)
    ax1.set_title("Window counts")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(block_counts.index.astype(str), block_counts.values)
    ax2.set_title("Feature count by block")
    ax2.set_ylabel("Number of features")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(True, axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(block_zero.index.astype(str), block_zero.values)
    ax3.set_title("Mean zero fraction by block")
    ax3.set_ylabel("Mean zero fraction")
    ax3.tick_params(axis="x", rotation=25)
    ax3.set_ylim(0, 1)
    ax3.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Combined feature overview — {title_suffix}", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def make_effect_heatmap(feature_catalog: pd.DataFrame, effects_A: pd.DataFrame, effects_B: pd.DataFrame, out_png: Path, title_suffix: str):
    eff = feature_catalog[["feature", "block"]].merge(
        effects_A[["feature", "cohens_d"]].rename(columns={"cohens_d": "A"}),
        on="feature", how="left"
    ).merge(
        effects_B[["feature", "cohens_d"]].rename(columns={"cohens_d": "B"}),
        on="feature", how="left"
    )

    eff["max_abs"] = eff[["A", "B"]].abs().max(axis=1)
    top = eff.sort_values(["max_abs", "block", "feature"], ascending=[False, True, True]).head(TOP_N_HEATMAP).copy()
    heat = top.set_index("feature")[["A", "B"]]
    vmax = float(np.nanmax(np.abs(heat.to_numpy(dtype=float))))
    vmax = max(vmax, 0.25)

    fig = plt.figure(figsize=(7.5, max(6.0, 0.22 * len(heat) + 1.6)), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    ax.set_yticks(np.arange(len(heat)), labels=heat.index.tolist())
    ax.set_xticks([0, 1], labels=["prdm9 vs datasetA", "prdm9 vs datasetB"])
    ax.set_title(f"Top feature effects (Cohen's d) — {title_suffix}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def make_block_effect_summary(feature_catalog: pd.DataFrame, effects_A: pd.DataFrame, effects_B: pd.DataFrame, out_png: Path, title_suffix: str):
    eff = feature_catalog[["feature", "block"]].merge(
        effects_A[["feature", "abs_cohens_d"]].rename(columns={"abs_cohens_d": "A"}),
        on="feature", how="left"
    ).merge(
        effects_B[["feature", "abs_cohens_d"]].rename(columns={"abs_cohens_d": "B"}),
        on="feature", how="left"
    )

    summary = eff.groupby("block", observed=True)[["A", "B"]].mean().reindex(BLOCK_ORDER).dropna(how="all")
    x = np.arange(len(summary))

    fig = plt.figure(figsize=(8, 4.8), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    w = 0.38
    ax.bar(x - w/2, summary["A"].values, width=w, label="prdm9 vs datasetA")
    ax.bar(x + w/2, summary["B"].values, width=w, label="prdm9 vs datasetB")
    ax.set_xticks(x, labels=summary.index.astype(str).tolist(), rotation=25, ha="right")
    ax.set_ylabel("Mean absolute Cohen's d")
    ax.set_title(f"Block-level discriminative strength — {title_suffix}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def make_pca_figure(all_df: pd.DataFrame, feature_catalog: pd.DataFrame, effects_A: pd.DataFrame, effects_B: pd.DataFrame, out_png: Path, title_suffix: str):
    top_dense = feature_catalog.sort_values("nonzero_frac_all", ascending=False).head(250)["feature"].tolist()
    top_effect = pd.concat([
        effects_A.sort_values("abs_cohens_d", ascending=False).head(250),
        effects_B.sort_values("abs_cohens_d", ascending=False).head(250),
    ], ignore_index=True)["feature"].tolist()

    selected = list(dict.fromkeys(top_dense + top_effect))[:500]
    X = all_df[selected].to_numpy(dtype=np.float64)
    Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(Xz)

    fig = plt.figure(figsize=(10.5, 4.5), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    for ax, bg in zip([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], ["datasetA_matched", "datasetB_global"]):
        sub_mask = all_df["WindowSet"].isin(["prdm9", bg]).to_numpy()
        sub_df = all_df.loc[sub_mask]
        sub_Z = Z[sub_mask]

        bg_mask = (sub_df["WindowSet"] == bg).to_numpy()
        pr_mask = (sub_df["WindowSet"] == "prdm9").to_numpy()

        ax.scatter(sub_Z[bg_mask, 0], sub_Z[bg_mask, 1], s=8, alpha=0.22, label=bg)
        ax.scatter(sub_Z[pr_mask, 0], sub_Z[pr_mask, 1], s=9, alpha=0.35, label="prdm9")
        ax.set_title(f"PCA: prdm9 vs {bg}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

    fig.suptitle(
        f"PCA on selected feature subset ({len(selected)} features) — {title_suffix}\n"
        f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def validate_combined_dataset(df: pd.DataFrame, feature_cols: list[str], expected_windows: pd.DataFrame):
    if len(df) != len(expected_windows):
        raise ValueError(f"Combined dataset row mismatch: {len(df)} vs expected {len(expected_windows)}")

    if df["window_uid"].duplicated().any():
        raise ValueError("Combined dataset has duplicate canonical window IDs")

    expected_uid = expected_windows["window_uid"].tolist()
    observed_uid = df["window_uid"].tolist()
    if expected_uid != observed_uid:
        raise ValueError("Combined dataset row order diverged from canonical window manifest")

    if df[feature_cols].isna().any().any():
        bad = df.loc[df[feature_cols].isna().any(axis=1), ["window_uid"] + ID_COLS].head()
        raise ValueError(f"Combined dataset contains missing feature values:\n{bad}")

    finite_mask = np.isfinite(df[feature_cols].to_numpy(dtype=np.float64))
    if not finite_mask.all():
        raise ValueError("Combined dataset contains non-finite feature values")

    if (df["window_length_bp"] <= 0).any():
        raise ValueError("Non-positive window lengths detected")

def make_feature_block_manifest(block_meta_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(block_meta_rows).sort_values("block_name", kind="mergesort").reset_index(drop=True)

# %% cell 3

# ------------------------------------------------------------------
# Build canonical window universe
# ------------------------------------------------------------------
logger.info("[1/5] Building canonical window manifest")
windows = build_window_manifest()
windows.to_csv(TABDIR / "window_manifest.tsv", sep="\t", index=False)

window_tbl = Table(title="Window manifest")
for ws in WINDOWSET_ORDER:
    n = int((windows["WindowSet"] == ws).sum())
    window_tbl.add_row(ws, f"{n:,}")
console.print(window_tbl)

logger.info("Canonical window manifest written: %s", TABDIR / "window_manifest.tsv")
logger.info("Total windows: %d", len(windows))

# %% cell 4

# ------------------------------------------------------------------
# Load and normalize all feature blocks
# ------------------------------------------------------------------
logger.info("[2/5] Loading and normalizing feature blocks")

normalized_blocks = {}
block_feature_cols = {}
block_meta_rows = []

for block_name, block_path in BLOCK_PATHS.items():
    block_df, feat_cols, meta = normalize_feature_block(block_path, block_name, windows)
    normalized_blocks[block_name] = block_df
    block_feature_cols[block_name] = feat_cols
    block_meta_rows.append(meta)

block_manifest = make_feature_block_manifest(block_meta_rows)
block_manifest.to_csv(TABDIR / "feature_block_manifest.tsv", sep="\t", index=False)
logger.info("Feature block manifest written: %s", TABDIR / "feature_block_manifest.tsv")

display(block_manifest)

# %% cell 5

# ------------------------------------------------------------------
# Assemble combined datasets for both G4 variants
# ------------------------------------------------------------------
logger.info("[3/5] Combining feature families into analysis-ready datasets")

variant_specs = {
    "variant_g4_detailed": ["g4_detailed", "methylation", "motifs", "accessibility", "histones", "recombination"],
    "variant_g4_collapsed": ["g4_collapsed", "methylation", "motifs", "accessibility", "histones", "recombination"],
}

run_metadata = {
    "n_jobs": N_JOBS,
    "window_manifest": str(TABDIR / "window_manifest.tsv"),
    "feature_block_manifest": str(TABDIR / "feature_block_manifest.tsv"),
    "block_sources": {k: str(v) for k, v in BLOCK_PATHS.items()},
    "variants": {},
    "notes": {
        "canonical_join_key": "window_uid = WindowSet::Chromosome:Start-End",
        "window_id_standardization": "Upstream blocks use mixed window_id types; final combined datasets standardize on canonical string window_uid plus stable integer window_index_global.",
        "merge_policy": "No silent zero-fill for merge failures; every block must cover the full canonical window universe exactly once.",
    },
}

for variant_name, use_blocks in variant_specs.items():
    logger.info("Combining variant: %s", variant_name)
    dirs = make_variant_outdirs(variant_name)

    merged = windows.copy()
    feature_cols = []
    used_meta = []

    for block_name in use_blocks:
        block_df = normalized_blocks[block_name]
        block_features = block_feature_cols[block_name]
        collision = sorted(set(feature_cols) & set(block_features))
        if collision:
            raise ValueError(f"[{variant_name}] feature name collision while adding {block_name}: {collision[:10]}")

        merged = merged.merge(
            block_df[["window_uid"] + block_features],
            on="window_uid",
            how="left",
            validate="one_to_one",
        )
        feature_cols.extend(block_features)
        used_meta.append(block_name)

    validate_combined_dataset(merged, feature_cols, windows)

    all_windows_path = dirs["features"] / "combined_all_windows.parquet"
    merged.to_parquet(all_windows_path, index=False)
    logger.info("[%s] wrote combined all-window dataset: %s", variant_name, all_windows_path)

    feature_catalog = parallel_feature_catalog(merged, feature_cols)
    effects_A = compute_comparison_effects(merged, feature_cols, "datasetA_matched")
    effects_B = compute_comparison_effects(merged, feature_cols, "datasetB_global")

    catalog = feature_catalog.merge(
        effects_A[["feature", "cohens_d", "abs_cohens_d", "log2_mean_ratio", "log2_prevalence_ratio"]].rename(columns={
            "cohens_d": "cohens_d_prdm9_vs_datasetA",
            "abs_cohens_d": "abs_cohens_d_prdm9_vs_datasetA",
            "log2_mean_ratio": "log2_mean_ratio_prdm9_vs_datasetA",
            "log2_prevalence_ratio": "log2_prevalence_ratio_prdm9_vs_datasetA",
        }),
        on="feature", how="left"
    ).merge(
        effects_B[["feature", "cohens_d", "abs_cohens_d", "log2_mean_ratio", "log2_prevalence_ratio"]].rename(columns={
            "cohens_d": "cohens_d_prdm9_vs_datasetB",
            "abs_cohens_d": "abs_cohens_d_prdm9_vs_datasetB",
            "log2_mean_ratio": "log2_mean_ratio_prdm9_vs_datasetB",
            "log2_prevalence_ratio": "log2_prevalence_ratio_prdm9_vs_datasetB",
        }),
        on="feature", how="left"
    )

    block_summary = (
        catalog.groupby("block", observed=True)
        .agg(
            n_features=("feature", "count"),
            mean_zero_frac=("zero_frac_all", "mean"),
            mean_abs_d_A=("abs_cohens_d_prdm9_vs_datasetA", "mean"),
            mean_abs_d_B=("abs_cohens_d_prdm9_vs_datasetB", "mean"),
        )
        .reindex(BLOCK_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    feature_catalog_path = dirs["tables"] / "feature_catalog.tsv"
    effects_A_path = dirs["tables"] / "effects_prdm9_vs_datasetA.tsv"
    effects_B_path = dirs["tables"] / "effects_prdm9_vs_datasetB.tsv"
    block_summary_path = dirs["tables"] / "block_summary.tsv"

    catalog.to_csv(feature_catalog_path, sep="\t", index=False)
    effects_A.to_csv(effects_A_path, sep="\t", index=False)
    effects_B.to_csv(effects_B_path, sep="\t", index=False)
    block_summary.to_csv(block_summary_path, sep="\t", index=False)

    ml_A = build_comparison_dataset(merged, "datasetA_matched", feature_cols)
    ml_B = build_comparison_dataset(merged, "datasetB_global", feature_cols)

    ml_A_path = dirs["features"] / "prdm9_vs_datasetA_matched.ml_ready.parquet"
    ml_B_path = dirs["features"] / "prdm9_vs_datasetB_global.ml_ready.parquet"
    enr_A_path = dirs["features"] / "prdm9_vs_datasetA_matched.enrichment_ready.parquet"
    enr_B_path = dirs["features"] / "prdm9_vs_datasetB_global.enrichment_ready.parquet"

    ml_A.to_parquet(ml_A_path, index=False)
    ml_B.to_parquet(ml_B_path, index=False)
    ml_A.to_parquet(enr_A_path, index=False)
    ml_B.to_parquet(enr_B_path, index=False)

    fig_overview = dirs["figures"] / "figure01_overview.png"
    fig_heatmap = dirs["figures"] / "figure02_top_feature_effects_heatmap.png"
    fig_block = dirs["figures"] / "figure03_block_effect_summary.png"
    fig_pca = dirs["figures"] / "figure04_pca_comparisons.png"

    make_overview_figure(merged, catalog, fig_overview, title_suffix=variant_name.replace("variant_", ""))
    make_effect_heatmap(catalog, effects_A, effects_B, fig_heatmap, title_suffix=variant_name.replace("variant_", ""))
    make_block_effect_summary(catalog, effects_A, effects_B, fig_block, title_suffix=variant_name.replace("variant_", ""))
    make_pca_figure(merged, catalog, effects_A, effects_B, fig_pca, title_suffix=variant_name.replace("variant_", ""))

    metadata = {
        "variant_name": variant_name,
        "g4_schema": "detailed" if "detailed" in variant_name else "collapsed",
        "used_blocks": used_meta,
        "n_windows_total": int(len(merged)),
        "n_features_total": int(len(feature_cols)),
        "n_windows_by_set": {k: int(v) for k, v in merged["WindowSet"].value_counts().to_dict().items()},
        "output_files": {
            "all_windows": str(all_windows_path),
            "ml_ready_datasetA": str(ml_A_path),
            "ml_ready_datasetB": str(ml_B_path),
            "enrichment_ready_datasetA": str(enr_A_path),
            "enrichment_ready_datasetB": str(enr_B_path),
            "feature_catalog": str(feature_catalog_path),
            "effects_datasetA": str(effects_A_path),
            "effects_datasetB": str(effects_B_path),
            "block_summary": str(block_summary_path),
            "overview_figure": str(fig_overview),
            "top_effect_heatmap": str(fig_heatmap),
            "block_effect_figure": str(fig_block),
            "pca_figure": str(fig_pca),
        },
    }

    with open(dirs["metadata"] / "variant_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    run_metadata["variants"][variant_name] = metadata

    summary_tbl = Table(title=f"{variant_name} summary")
    summary_tbl.add_row("All windows", f"{len(merged):,}")
    summary_tbl.add_row("Features", f"{len(feature_cols):,}")
    summary_tbl.add_row("ML-ready A", str(ml_A_path))
    summary_tbl.add_row("ML-ready B", str(ml_B_path))
    summary_tbl.add_row("Catalog", str(feature_catalog_path))
    console.print(summary_tbl)

with open(METADIR / "combine_run_metadata.json", "w") as f:
    json.dump(run_metadata, f, indent=2)

logger.info("Top-level run metadata written: %s", METADIR / "combine_run_metadata.json")

# %% cell 6

# ------------------------------------------------------------------
# Final concise report
# ------------------------------------------------------------------
report = Table(title="Combined production datasets")
report.add_column("Variant")
report.add_column("Windows", justify="right")
report.add_column("Features", justify="right")
report.add_column("Main ML-ready outputs")

for variant_name, meta in run_metadata["variants"].items():
    report.add_row(
        variant_name,
        f"{meta['n_windows_total']:,}",
        f"{meta['n_features_total']:,}",
        f"{meta['output_files']['ml_ready_datasetA']}\n{meta['output_files']['ml_ready_datasetB']}",
    )

console.print(report)
console.print(f"[bold green]Done.[/bold green] Log file: {LOGDIR / 'combine_run.log'}")
console.print(f"[bold green]Run metadata:[/bold green] {METADIR / 'combine_run_metadata.json'}")
