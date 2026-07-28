#!/usr/bin/env python3
"""
Chromatin Accessibility-G4 Overlap Analysis

Question: Do open chromatin regions overlap with G-quadruplex (G4) structures?

Usage:
    python analyze_accessibility_g4.py [--base-dir BASE_DIR] [--output-dir OUTPUT_DIR]

Example:
    python analyze_accessibility_g4.py --base-dir /path/to/data --output-dir ./results
"""

import json, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
from matplotlib.patches import Patch

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    # Fallback if rich not installed
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

def setup_nature_style():
    plt.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 600,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'font.size': 10,
        'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9, 'font.family': 'sans-serif',
        'pdf.fonttype': 42,
    })

COLORS = {'no_g4': '#BDC3C7', 'has_g4': '#E74C3C', 'text': '#2C3E50'}
G4_COLORS = {
    'experimental': '#C44E52', 'short': '#4C72B0', 'median': '#55A868',
    'long': '#DD8452', 'noncanonical': '#9468B4',
}

DATASET_TITLES = {
    'prdm9': 'PRDM9-Binding',
    'datasetA_matched': 'DatasetA Matched',
    'datasetB_global': 'DatasetB Global',
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_g4_dicts(g4_dir):
    console.print("\n[bold]Loading G4 intervals...[/bold]")
    
    def load_bed(file, header=False):
        if header:
            df = pd.read_csv(file, sep='\t', header=0)
            df['Start'] = df['start'] - 1
            df['End'] = df['end']
            return df[['seqnames','Start','End']].rename(columns={'seqnames':'chrom'})
        else:
            return pd.read_csv(file, sep='\t', header=None, names=['chrom','Start','End'])
    
    g4_dict = {}
    
    exp = load_bed(g4_dir / "HEK293_G4s_hg38.bed", header=False)
    g4_dict['experimental'] = {c: g[['Start','End']].values for c, g in exp.groupby('chrom')}
    console.print(f"  Experimental: {len(exp):,}")
    
    for name, file in [('short', 'G4s_short_pqsfinder_hg38.txt'), 
                       ('median', 'G4s_median_pqsfinder_hg38.txt'), 
                       ('long', 'G4s_long_pqsfinder_hg38.txt')]:
        df = load_bed(g4_dir / file, header=True)
        g4_dict[name] = {c: g[['Start','End']].values for c, g in df.groupby('chrom')}
        console.print(f"  {name.capitalize()}: {len(df):,}")
    
    noncan = load_bed(g4_dir / "G4s_noncanonical_pqsfinder_hg38.bed", header=False)
    g4_dict['noncanonical'] = {c: g[['Start','End']].values for c, g in noncan.groupby('chrom')}
    console.print(f"  Noncanonical: {len(noncan):,}")
    
    return g4_dict

# ============================================================================
# ANALYSIS
# ============================================================================

def has_overlap(ws, we, intervals):
    if len(intervals) == 0:
        return False
    return ((intervals[:,1] > ws) & (intervals[:,0] < we)).any()

def analyze_dataset(name, windows_df, g4_dict):
    console.print(f"\n[bold]{name}[/bold] ({len(windows_df):,} windows)")
    
    results = []
    for chrom in windows_df['Chromosome'].unique():
        chrom_df = windows_df[windows_df['Chromosome'] == chrom]
        for idx, row in chrom_df.iterrows():
            acc = row['accessibility__mean']
            ws, we = row['Start'], row['End']
            entry = {'window_id': row['window_id'], 'accessibility': acc}
            
            for g4_type in ['experimental', 'short', 'median', 'long', 'noncanonical']:
                if chrom in g4_dict[g4_type]:
                    entry[f'{g4_type}_has'] = 1 if has_overlap(ws, we, g4_dict[g4_type][chrom]) else 0
                else:
                    entry[f'{g4_type}_has'] = 0
            results.append(entry)
    
    result_df = pd.DataFrame(results)
    console.print(f"  ✓ Classified {len(result_df):,} windows")
    return result_df

def calc_stats(result_df, g4_type):
    has_col = f'{g4_type}_has'
    no_g4 = result_df[result_df[has_col]==0]['accessibility']
    has_g4 = result_df[result_df[has_col]==1]['accessibility']
    
    if len(has_g4) == 0:
        return None
    
    stat, pval = stats.mannwhitneyu(no_g4, has_g4, alternative='less')
    n1, n2 = len(no_g4), len(has_g4)
    effect = 1 - (2 * stat) / (n1 * n2) if n1 * n2 > 0 else 0
    
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    pval_text = 'p<0.001' if pval < 0.001 else f'p={pval:.2e}' if pval < 0.01 else f'p={pval:.3f}'
    
    return {
        'n_no': n1, 'n_has': n2,
        'median_no': float(no_g4.median()), 'median_has': float(has_g4.median()),
        'median_ratio': float(has_g4.median() / no_g4.median()) if no_g4.median() > 0 else None,
        'effect_size': float(effect), 'sig': sig, 'pval': float(pval), 'pval_text': pval_text
    }

# ============================================================================
# PLOTTING
# ============================================================================

def plot_violin(result_df, name, out_file, stats_dict):
    setup_nature_style()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    g4_types = ['experimental', 'short', 'median', 'long', 'noncanonical']
    g4_labels = ['Experimental', 'Short', 'Median', 'Long', 'Noncanonical']
    
    # Y-axis limits
    if name == 'datasetB_global':
        y_min, y_max = 0, 50
    else:
        all_acc = result_df['accessibility'].values
        y_max = np.percentile(all_acc, 99)
        y_min = min(0, np.percentile(all_acc, 1))
    
    for i, (g4_type, g4_label) in enumerate(zip(g4_types, g4_labels)):
        ax = axes[i]
        has_col = f'{g4_type}_has'
        
        no_g4 = result_df[result_df[has_col]==0]['accessibility'].values
        has_g4 = result_df[result_df[has_col]==1]['accessibility'].values
        
        # Violins
        vp = ax.violinplot([no_g4, has_g4], positions=[1, 2], widths=0.65,
                           showmeans=False, showmedians=False, showextrema=True)
        
        for pc, color in zip(vp['bodies'], [COLORS['no_g4'], COLORS['has_g4']]):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.85)
            pc.set_linewidth(0.8)
        
        # Medians
        for data, pos, mcolor in [(no_g4, 1, 'black'), (has_g4, 2, 'white')]:
            ax.plot(pos, np.median(data), 'o', markersize=8, color=mcolor,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        
        # Quartiles
        for data, pos in [(no_g4, 1), (has_g4, 2)]:
            q1, q3 = np.percentile(data, [25, 75])
            ax.plot([pos-0.08, pos+0.08], [q1, q1], 'k-', linewidth=1.5, alpha=0.6)
            ax.plot([pos-0.08, pos+0.08], [q3, q3], 'k-', linewidth=1.5, alpha=0.6)
        
        # Axes
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No G4', 'Has G4'], fontsize=10, fontweight='bold')
        ax.set_title(g4_label, fontsize=11, fontweight='bold', pad=12, color=G4_COLORS[g4_type])
        
        if i == 0:
            ax.set_ylabel('Chromatin Accessibility\n(mean signal)', fontsize=10, fontweight='bold')
        
        ax.set_ylim(y_min, y_max)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        # N values
        ax.text(1.35, y_max * 0.95, f'n={len(no_g4):,}', ha='left', va='top',
               fontsize=9, fontweight='bold', color=COLORS['text'])
        ax.text(2.35, y_max * 0.95, f'n={len(has_g4):,}', ha='left', va='top',
               fontsize=9, fontweight='bold', color=COLORS['text'])
        
        # Statistics
        if g4_type in stats_dict:
            s = stats_dict[g4_type]
            ax.text(1.5, y_max * 0.82, s['sig'], ha='center', va='top',
                   fontsize=18, fontweight='bold', color='black')
            ax.text(1.5, y_max * 0.70, f'r={s["effect_size"]:.2f}',
                   ha='center', va='top', fontsize=8, fontweight='bold', color=G4_COLORS[g4_type])
    
    # Title
    title = DATASET_TITLES.get(name, name.replace('_', ' ').title())
    fig.suptitle(f'{title}', fontsize=14, fontweight='bold', y=0.99, color=COLORS['text'])
    
    # X-axis label
    fig.text(0.5, 0.03, 'G4 Overlap Status', ha='center', fontsize=12, fontweight='bold', color=COLORS['text'])
    
    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['no_g4'], edgecolor='black', label='No G4'),
        Patch(facecolor=COLORS['has_g4'], edgecolor='black', label='Has G4'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.97),
              ncol=2, frameon=False, fontsize=9)
    
    # Bottom annotation
    stats_row1 = []
    stats_pval = []
    for g4_type in g4_types:
        if g4_type in stats_dict:
            s = stats_dict[g4_type]
            label = g4_type[:4].upper()
            stats_row1.append(f"{label}: {s['median_ratio']:.2f}×")
            stats_pval.append(f"{label}: {s['pval_text']}")
    
    row1_text = '  |  '.join(stats_row1)
    row2_text = '  |  '.join(stats_pval)
    row3_text = 'Fold-change: median accessibility (Has G4 / No G4) | p-value: Mann-Whitney U test (one-tailed) | r: effect size (0.1=small, 0.3=medium, 0.5=large)'
    
    fig.text(0.5, -0.04, row1_text, ha='center', va='top', fontsize=8, color=COLORS['text'], fontweight='bold')
    fig.text(0.5, -0.07, row2_text, ha='center', va='top', fontsize=8, color=COLORS['text'], fontweight='bold')
    fig.text(0.5, -0.10, row3_text, ha='center', va='top', fontsize=7, color=COLORS['text'], style='italic')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(out_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Chromatin Accessibility-G4 Overlap Analysis')
    parser.add_argument('--base-dir', type=str, default='/home/jovyan/prdm9_new',
                       help='Base directory containing input data')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for results')
    args = parser.parse_args()
    
    start = datetime.now()
    console.print(Panel.fit(f"[bold]Accessibility-G4 Analysis[/bold]\nOutput: {args.output_dir}"))
    
    # Setup paths
    base = Path(args.base_dir)
    outdir = Path(args.output_dir)
    for p in [outdir, outdir/'figures', outdir/'tables', outdir/'metadata']:
        p.mkdir(parents=True, exist_ok=True)
    
    # Load G4s
    g4_dict = load_g4_dicts(base / 'g4')
    
    # Load accessibility
    console.print("\n[bold]Loading accessibility data...[/bold]")
    acc_file = base / "data_prdm9_accessibility_production/features/accessibility_features.parquet"
    acc = pd.read_parquet(acc_file)
    console.print(f"  ✓ {len(acc):,} windows")
    
    # Analyze
    console.print("\n[bold]Analyzing G4 overlaps...[/bold]")
    all_results = {}
    all_stats = {}
    
    for name in ['prdm9', 'datasetA_matched', 'datasetB_global']:
        windows = acc[acc['WindowSet']==name].copy()
        result_df = analyze_dataset(name, windows, g4_dict)
        all_results[name] = result_df
        
        stats_dict = {gt: calc_stats(result_df, gt) 
                     for gt in ['experimental', 'short', 'median', 'long', 'noncanonical']}
        all_stats[name] = stats_dict
        
        # Save tables
        result_df.to_csv(outdir/'tables'/f"acc_g4_{name}.tsv", sep='\t', index=False)
        with open(outdir/'metadata'/f"stats_{name}.json", 'w') as f:
            json.dump(stats_dict, f, indent=2)
        console.print(f"  ✓ Saved: {name}")
    
    # Plot
    console.print("\n[bold]Generating figures...[/bold]")
    for name, result_df in all_results.items():
        fig_file = outdir/'figures'/f"accessibility_g4_{name}.pdf"
        plot_violin(result_df, name, fig_file, all_stats[name])
        console.print(f"  ✓ {fig_file.name}")
    
    # Summary
    with open(outdir/'metadata'/'analysis.json', 'w') as f:
        json.dump({'start': start.isoformat(), 'end': datetime.now().isoformat(),
                  'datasets': list(all_results.keys())}, f, indent=2)
    
    duration = datetime.now() - start
    console.print(Panel(f"[bold green]✓ Complete[/bold green]\nDuration: {duration}"))

if __name__ == '__main__':
    main()
