#!/usr/bin/env python3
"""
Step 2: Motif-G4 Overlap Analysis

Scans for PRDM9 motif hits and calculates overlaps with G4 structures.

Inputs:
    - hg38.fa: Reference genome
    - motifs/motifs.txt: PRDM9 MEME motifs
    - g4/HEK293_G4s_hg38.bed: Experimental G4s
    - g4/G4s_{short,median,long}_pqsfinder_hg38.txt: Canonical G4s
    - g4/G4s_noncanonical_pqsfinder_hg38.bed: Noncanonical G4s (from Step 1)
    - data_prdm9_sampling_production/*.bed: Window sets

Outputs:
    - figures/motif_g4_overlap_{dataset}.pdf: Publication figures
    - tables/overlap_counts_{dataset}.tsv: Overlap count tables
    - metadata/analysis_metadata.json: Run metadata

Runtime: ~25 minutes (28 cores, 424K windows)
"""

import os
import sys

# Prevent OpenMP oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pysam
from numba import njit
from rich.console import Console
from rich.panel import Panel

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = Path(".")
FASTA = BASE / "hg38.fa"
MEME = BASE / "motifs" / "motifs.txt"
G4_DIR = BASE / "g4"
SAMPLING = BASE / "data_prdm9_sampling_production"
OUTDIR = BASE / "data_prdm9_motif_g4_overlap_analysis"

# Create output directories
for p in [OUTDIR/"figures", OUTDIR/"tables", OUTDIR/"metadata", OUTDIR/"logs"]:
    p.mkdir(parents=True, exist_ok=True)

# Analysis parameters
PSEUDO = 1e-6
BG = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
N_CPUS = min(28, cpu_count() or 1)
HIT_THRESHOLD = 0.0  # Log-odds score threshold
MOTIFS = [f"Human{i}" for i in range(1, 18)]

# Dataset configuration
DATASETS = [
    ('prdm9', 'PRDM9-binding dataset', SAMPLING/"prdm9_filtered.bed"),
    ('datasetA_matched', 'DatasetA Matched', SAMPLING/"datasetA_matched.len_gc_entropy.bed"),
    ('datasetB_global', 'DatasetB Global', SAMPLING/"datasetB_global.length_only.bed")
]

# ============================================================================
# MOTIF LOADING
# ============================================================================

def load_meme(path):
    """Load MEME motif PWMs."""
    motifs, current = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('MOTIF '):
                current = line.split()[1]
            elif current and line[0].isdigit():
                if current not in motifs:
                    motifs[current] = []
                motifs[current].append([float(x) for x in line.split()])
    return {k: np.array(v, dtype=np.float64) for k, v in motifs.items()}

# ============================================================================
# MOTIF SCANNING (Numba-optimized)
# ============================================================================

@njit
def scan_seq(seq_codes, pwm, threshold=HIT_THRESHOLD):
    """Scan sequence for motif hits using PWM."""
    hits, L = [], pwm.shape[0]
    for i in range(len(seq_codes) - L + 1):
        # Check for valid bases (A=0, C=1, G=2, T=3)
        valid = True
        for j in range(L):
            if seq_codes[i+j] < 0 or seq_codes[i+j] >= 4:
                valid = False
                break
        if not valid:
            continue
        
        # Calculate log-odds score
        score = 0.0
        for j in range(L):
            if seq_codes[i+j] >= 0:
                score += np.log2((pwm[j, seq_codes[i+j]] + PSEUDO) / 
                                (BG[seq_codes[i+j]] + PSEUDO))
        
        if score >= threshold:
            hits.append((i, i + L, score))
    return hits


def encode(seq):
    """Encode DNA sequence as numeric array."""
    return np.array([{'A':0,'C':1,'G':2,'T':3}.get(b.upper(), -1) 
                     for b in seq], dtype=np.int8)


def rc(seq):
    """Get reverse complement of DNA sequence."""
    return ''.join({'A':'T','T':'A','C':'G','G':'C','N':'N'}.get(b.upper(), 'N') 
                   for b in reversed(seq))

# ============================================================================
# OVERLAP DETECTION (Numba-optimized)
# ============================================================================

@njit
def count_overlap(m_starts, m_ends, g_starts, g_ends):
    """
    Count overlaps using two-pointer sweep algorithm.
    
    Complexity: O(n + m) where n=len(motif hits), m=len(G4s)
    Overlap definition: any base-pair overlap (>=1 bp)
    """
    if len(m_starts) == 0 or len(g_starts) == 0:
        return 0
    
    mi, gi, cnt = 0, 0, 0
    ms = np.argsort(m_starts)
    gs = np.argsort(g_starts)
    
    while mi < len(m_starts) and gi < len(g_starts):
        ms_s = m_starts[ms[mi]]
        ms_e = m_ends[ms[mi]]
        gs_s = g_starts[gs[gi]]
        gs_e = g_ends[gs[gi]]
        
        # Check for overlap
        if ms_e > gs_s and ms_s < gs_e:
            cnt += 1
            if ms_e < gs_e:
                mi += 1
            else:
                gi += 1
        elif ms_e <= gs_s:
            mi += 1
        else:
            gi += 1
    
    return cnt

# ============================================================================
# DATA LOADING
# ============================================================================

def load_g4():
    """Load all G4 categories."""
    c = Console()
    c.print("[bold]Loading G4 data...[/bold]")
    
    # Experimental G4s
    exp = pd.read_csv(G4_DIR/"HEK293_G4s_hg38.bed", sep='\t', header=None, 
                      names=['chrom','start','end'])
    exp_dict = {chrom: (df['start'].values.astype(np.int64), 
                        df['end'].values.astype(np.int64))
                for chrom, df in exp.groupby('chrom')}
    c.print(f"  Experimental: {len(exp):,}")
    
    # Four sequence categories
    seq_dict = {}
    files = [
        ('short', 'G4s_short_pqsfinder_hg38.txt'),
        ('median', 'G4s_median_pqsfinder_hg38.txt'),
        ('long', 'G4s_long_pqsfinder_hg38.txt'),
        ('noncanonical', 'G4s_noncanonical_pqsfinder_hg38.bed')
    ]
    
    for sc, fname in files:
        if 'noncanonical' in fname:
            df = pd.read_csv(G4_DIR/fname, sep='\t', header=None, 
                            names=['chrom','start','end'])
        else:
            df = pd.read_csv(G4_DIR/fname, sep='\t', header=0)
            df = df[['seqnames','start','end']].rename(columns={'seqnames':'chrom'})
        
        seq_dict[sc] = {chrom: (df_group['start'].values.astype(np.int64),
                                df_group['end'].values.astype(np.int64))
                       for chrom, df_group in df.groupby('chrom')}
        c.print(f"  {sc.capitalize()}: {len(df):,}")
    
    return exp_dict, seq_dict


def load_windows(bed, name):
    """Load genomic windows from BED file."""
    df = pd.read_csv(bed, sep='\t', header=None, 
                     names=['chrom','start','end'])
    return [(f"{name}::{row.chrom}:{row.start}-{row.end}", 
             row.chrom, row.start, row.end) 
            for row in df.itertuples()]

# ============================================================================
# PARALLEL WORKER
# ============================================================================

def worker(args):
    """Process a chunk of windows in parallel."""
    chunks, fasta_path, motifs, exp_dict, seq_dict = args
    fasta = pysam.FastaFile(str(fasta_path))
    res_exp, res_seq = [], []
    
    for wid, chrom, start, end in chunks:
        # Extract sequence
        try:
            seq = fasta.fetch(chrom, start, end)
        except Exception:
            continue
        
        codes = encode(seq.upper())
        wlen = len(seq)
        
        # Scan all motifs on both strands
        hits = {}
        for mn, pwm in motifs.items():
            fwd = scan_seq(codes, pwm)
            rev = [(wlen-h[1], wlen-h[0], h[2]) 
                   for h in scan_seq(encode(rc(seq.upper())), pwm)]
            
            # Deduplicate by start position (keep highest score)
            merged = {}
            for h in fwd + rev:
                if h[0] not in merged or h[2] > merged[h[0]][2]:
                    merged[h[0]] = h
            
            # Convert to genomic coordinates
            hits[mn] = [(start+h[0], start+h[1], h[2]) for h in merged.values()]
        
        # Calculate overlaps with experimental G4s
        o_exp = {}
        if chrom in exp_dict:
            gs, ge = exp_dict[chrom]
            for mn, h in hits.items():
                if not h:
                    o_exp[mn] = 0
                else:
                    ms = np.array([x[0] for x in h], dtype=np.int64)
                    me = np.array([x[1] for x in h], dtype=np.int64)
                    o_exp[mn] = count_overlap(ms, me, gs, ge)
        else:
            o_exp = {m: 0 for m in hits}
        
        res_exp.append({'wid': wid, 'chrom': chrom, 'start': start, 'end': end, **o_exp})
        
        # Calculate overlaps with sequence G4s (4 categories)
        o_seq = {}
        for sc in ['short', 'median', 'long', 'noncanonical']:
            if chrom in seq_dict.get(sc, {}):
                gs, ge = seq_dict[sc][chrom]
                for mn, h in hits.items():
                    if not h:
                        o_seq[f'{mn}_{sc}'] = 0
                    else:
                        ms = np.array([x[0] for x in h], dtype=np.int64)
                        me = np.array([x[1] for x in h], dtype=np.int64)
                        o_seq[f'{mn}_{sc}'] = count_overlap(ms, me, gs, ge)
            else:
                for mn in hits:
                    o_seq[f'{mn}_{sc}'] = 0
        
        res_seq.append({'wid': wid, 'chrom': chrom, 'start': start, 'end': end, **o_seq})
    
    fasta.close()
    return res_exp, res_seq

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot(exp_sums, seq_sums, name, display_name, out):
    """Generate publication-quality figure."""
    mpl.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 600,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    
    colors = {'short': "#4C72B0", 'median': "#55A868", 
              'long': "#DD8452", 'noncanonical': "#9468B4"}
    
    # Panel 1: Experimental
    ax1 = axes[0]
    vals = [exp_sums.get(m, 0) for m in MOTIFS]
    x = np.arange(len(MOTIFS))
    bars1 = ax1.bar(x, vals, color="#C44E52", width=0.7, 
                    edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('PRDM9 Motif', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Motif Hits Overlapping G4s', fontsize=10, fontweight='bold')
    ax1.set_title('Experimental G4-seq', fontsize=11, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MOTIFS, rotation=45, ha='right')
    ax1.set_xlim(-0.5, len(MOTIFS)-0.5)
    
    # Value labels (60° rotation)
    for bar, val in zip(bars1, vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + max(vals)*0.02,
                    f'{val:,}', ha='center', va='bottom', 
                    fontsize=6, rotation=60)
    
    # Panel 2: Sequence (stacked)
    ax2 = axes[1]
    bottom = np.zeros(len(MOTIFS))
    
    for sc in ['short', 'median', 'long', 'noncanonical']:
        v = [seq_sums.get(f'{m}_{sc}', 0) for m in MOTIFS]
        ax2.bar(x, v, bottom=bottom, color=colors[sc], width=0.7, 
                edgecolor='black', linewidth=0.5, label=sc.capitalize())
        bottom += v
    
    ax2.set_xlabel('PRDM9 Motif', fontsize=10, fontweight='bold')
    ax2.set_title('Sequence Predictions (pqsfinder)', 
                  fontsize=11, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MOTIFS, rotation=45, ha='right')
    ax2.set_xlim(-0.5, len(MOTIFS)-0.5)
    
    # Legend (positioned below x-axis)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.65), 
               frameon=False, ncol=4, title='G4 Category', 
               title_fontsize=8, fontsize=8)
    
    # Synchronize Y-axis
    ymax = max(max(vals) if vals else 0, bottom.max() if len(bottom)>0 else 0) * 1.20
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)
    
    # Total labels (60° rotation)
    for i, total in enumerate(bottom):
        if total > 0:
            ax2.text(i, total + ymax*0.02, 
                    f'{int(total):,}', ha='center', va='bottom', 
                    fontsize=6, rotation=60)
    
    # Light grid lines
    ax1.axhline(y=ymax*0.5, color='gray', linestyle='--', 
                linewidth=0.3, alpha=0.3)
    ax2.axhline(y=ymax*0.5, color='gray', linestyle='--', 
                linewidth=0.3, alpha=0.3)
    
    # Main title
    fig.suptitle(f'{display_name}\nG4 category overlaps with PRDM9 motif', 
                 fontsize=12, fontweight='bold', y=1.08)
    
    plt.tight_layout(rect=[0, -0.05, 1, 0.98])
    plt.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete motif-G4 overlap analysis."""
    t0 = datetime.now()
    c = Console()
    
    c.print(Panel.fit(
        f"[bold]Motif-G4 Overlap Analysis[/bold]\n"
        f"Cores: {N_CPUS}\n"
        f"Output: {OUTDIR}"
    ))
    
    # Validate inputs
    c.print("\n[bold]Validating inputs...[/bold]")
    required = [FASTA, MEME, 
                G4_DIR/"HEK293_G4s_hg38.bed",
                G4_DIR/"G4s_short_pqsfinder_hg38.txt",
                G4_DIR/"G4s_median_pqsfinder_hg38.txt",
                G4_DIR/"G4s_long_pqsfinder_hg38.txt",
                G4_DIR/"G4s_noncanonical_pqsfinder_hg38.bed"]
    
    for f in required:
        if not f.exists():
            c.print(f"[red]ERROR: Missing {f}[/red]")
            sys.exit(1)
    c.print("  ✓ All inputs present")
    
    # Load motifs
    c.print("\n[bold]Loading motifs...[/bold]")
    motifs = load_meme(MEME)
    c.print(f"  ✓ {len(motifs)} motifs")
    
    # Load G4 data
    exp_dict, seq_dict = load_g4()
    
    # Process each dataset
    for file_id, display_name, bed in DATASETS:
        c.print(f"\n[bold]{display_name}[/bold]")
        
        # Load windows
        windows = load_windows(bed, file_id)
        c.print(f"  Windows: {len(windows):,}")
        
        # Create chunks for parallel processing
        chunk_size = max(100, len(windows) // (N_CPUS * 4))
        chunks = [windows[i:i+chunk_size] for i in range(0, len(windows), chunk_size)]
        args_list = [(chunk, FASTA, motifs, exp_dict, seq_dict) for chunk in chunks]
        
        c.print(f"  Processing: {len(chunks)} chunks ({N_CPUS} cores)")
        
        # Parallel processing
        all_exp, all_seq = [], []
        with Pool(N_CPUS) as pool:
            for res_exp, res_seq in tqdm(pool.imap(worker, args_list), 
                                         total=len(chunks), desc="  Progress"):
                all_exp.extend(res_exp)
                all_seq.extend(res_seq)
        
        # Aggregate results
        df_exp = pd.DataFrame(all_exp)
        df_seq = pd.DataFrame(all_seq)
        
        exp_sums = df_exp[MOTIFS].sum()
        seq_sums = {f'{m}_{sc}': df_seq[f'{m}_{sc}'].sum() 
                   for sc in ['short','median','long','noncanonical'] 
                   for m in MOTIFS}
        
        # Save table
        tbl = pd.DataFrame({
            'Motif': MOTIFS,
            'Experimental': [exp_sums.get(m, 0) for m in MOTIFS],
            'Short': [seq_sums.get(f'{m}_short', 0) for m in MOTIFS],
            'Median': [seq_sums.get(f'{m}_median', 0) for m in MOTIFS],
            'Long': [seq_sums.get(f'{m}_long', 0) for m in MOTIFS],
            'Noncanonical': [seq_sums.get(f'{m}_noncanonical', 0) for m in MOTIFS],
        })
        tbl['Total'] = tbl[['Short','Median','Long','Noncanonical']].sum(axis=1)
        tbl.to_csv(OUTDIR/f"tables/{file_id}.tsv", sep='\t', index=False)
        c.print(f"  ✓ Table saved")
        
        # Generate figure
        plot(exp_sums, seq_sums, file_id, display_name, 
             OUTDIR/f"figures/{file_id}.pdf")
        c.print(f"  ✓ Figure saved")
    
    # Save metadata
    metadata = {
        'timestamp': t0.isoformat(),
        'completion_time': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - t0).total_seconds(),
        'parameters': {
            'n_cpus': N_CPUS,
            'hit_threshold': HIT_THRESHOLD,
            'pseudo_count': PSEUDO
        },
        'datasets': [d[0] for d in DATASETS],
        'motifs': MOTIFS
    }
    
    import json
    with open(OUTDIR/'metadata/analysis_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    c.print(Panel(
        f"[bold green]✓ Analysis Complete[/bold green]\n"
        f"Duration: {datetime.now() - t0}"
    ))


if __name__ == '__main__':
    main()
