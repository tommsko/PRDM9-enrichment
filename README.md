# PRDM9 Binding Site Analysis

**PRDM9 uses G-quadruplex structures as binding platforms**

Computational pipeline for characterizing PRDM9 binding sites in HEK293 cells.

## Key Findings

- G4 motifs among most enriched features at PRDM9 binding sites
- Supported G4s show 71-fold enrichment over sequence-matched background
- PRDM9 directly binds G4 structures in vitro
- Canonical G4s show stronger enrichment than non-canonical

## Quick Start

```bash
# Clone repository
git clone https://github.com/tommsko/PRDM9-enrichment
cd PRDM9-enrichment

# Create environment (mamba recommended)
mamba env create -f environment.yml
mamba activate prdm9-analysis

# Or with pip
pip install -r requirements.txt
```

## Repository Structure

```
PRDM9-enrichment/
├── 1_sample_windows.ipynb       # Stage 1: Window sampling
├── 2[a-f]_feat_*.ipynb          # Stage 2: Feature generation (6 notebooks)
├── 2g_combine.ipynb             # Stage 3: Combine features into dataset
├── 3[a-c]_enrichment_*.ipynb    # Stage 4: Enrichment analysis (3 notebooks)
├── environment.yml              # Mamba/conda environment
├── requirements.txt             # Pip requirements
└── README.md
```

## Data Sources

| File                                                              | Size     | MD5 Checksum                       | Source               | Accession / Command                                                                                                              |
| ----------------------------------------------------------------- | -------- | ---------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Reference Genome**                                              |          |                                    |                      |                                                                                                                                  |
| hg38.fa                                                           | 3.05 GB  | `b2aee9f885accc00531e59c4736bee63` | UCSC                 | -                                                                                                                                |
| hg38.fa.fai                                                       | 19 KB    | `1954682fe412d4c5f8b93ad8e4083181` | -                    | TODO                                                                                                                             |
| **PRDM9 ChIP-seq**                                                |          |                                    |                      |                                                                                                                                  |
| prdm9_peaks/HEK293_PRDM9-B_HA_MACS2_hg38.bed                      | 1.44 MB  | `6dc00b4021b540eb0d504f45d54b4021` | GEO                  | GSE99407                                                                                                                         |
| **G4 Predictions (pqsfinder)**                                    |          |                                    |                      |                                                                                                                                  |
| g4/pqsfinder_hg38.bed                                             | 39.81 MB | `6bc2d5dceec34a9b593dd751a417c73a` | Hon et al. 2017      | TODO                                                                                                                             |
| g4/G4s_short_pqsfinder_hg38.txt                                   | 34.67 MB | `b1dc6251a23d80285027470b78711ee3` | pqsfinder subset     | TODO: Angelika's regex                                                                                                           |
| g4/G4s_median_pqsfinder_hg38.txt                                  | 39.99 MB | `7f5736526cf39a1babfd6e3520ab1118` | pqsfinder subset     | TODO: Angelika's regex                                                                                                           |
| g4/G4s_long_pqsfinder_hg38.txt                                    | 33.69 MB | `d2d5ab988620aaafd4fab61fc4dc4a6f` | pqsfinder subset     | TODO: Angelika's regex                                                                                                           |
| **G4 Experimental (CUT&Tag)**                                     |          |                                    |                      |                                                                                                                                  |
| g4/HEK293_G4s_hg38.bed                                            | 1 MB     | `29655117b69960051bacb390c801c33c` | SRA                  | SRR14300945, SRR14879748, SRR26244907, SRR26244908, SRR26244909, SRR26244910, SRR26244911, SRR26244912, SRR25010684, SRR25010685 |
| **DNA Methylation (RRBS)**                                        |          |                                    |                      |                                                                                                                                  |
| methylation/ENCFF001TMQ_HEK_RRBS_DNAmet_R1_hg38.bed               | 109 MB   | `3441dfc226422f3790dd83c8a3497839` | ENCODE               | ENCFF001TMQ                                                                                                                      |
| methylation/ENCFF001TMQ_HEK_RRBS_DNAmet_R1_hg38.without_track.bed | 109 MB   | `d9e4cc7aa5d03b5bce840cfcd2e80b4e` | ENCODE               | ENCFF001TMQ                                                                                                                      |
| methylation/ENCFF001TMR_HEK_RRBS_DNAmet_R2_hg38.bed               | 93 MB    | `677357ae1325d6a821c5b8d6f0c505e2` | ENCODE               | ENCFF001TMR                                                                                                                      |
| methylation/ENCFF001TMR_HEK_RRBS_DNAmet_R2_hg38.without_track.bed | 93 MB    | `ff369f69a140dc08b8a843a1381bbd84` | ENCODE               | ENCFF001TMR                                                                                                                      |
| methylation/HEK_RRBS_DNAmet_R1+R2_hg38.bed                        | 295 MB   | `ed70e83aa3b2c23e4aa9f51989de4dfd` | ENCODE               | TODO                                                                                                                             |
| methylation/HEK_RRBS_DNAmet_R1+R2_hg38.sorted.bed                 | 295 MB   | `b178a11f662c335d8203f36cec3d8e9b` | ENCODE               | TODO                                                                                                                             |
| **Sequence Motifs**                                               |          |                                    |                      |                                                                                                                                  |
| motifs/motifs.txt                                                 | 37 KB    | `76d36747cb7bb6042dd01f21c22199b2` | Altemose et al. 2017 | Figure 1—source data 2. PWMs for all motifs, in MEME format (elife-28383-fig1-data2.txt)                                         |
| **Chromatin Accessibility (ATAC-seq)**                            |          |                                    |                      |                                                                                                                                  |
| accessibility/ENCFF668ATM.bigWig                                  | 612 MB   | `0c1426af95a49f253599359224667995` | ENCODE               | ENCSR205FUM                                                                                                                      |
| **Meiotic Recombination Hotspots**                                |          |                                    |                      |                                                                                                                                  |
| meiotic_recombination/meiotic_recombination_DSB_map_hg38.bed      | 1.77 MB  | `b380b6d3382db7d966ce5cf6c93bca24` | TODO                 | TODO                                                                                                                             |
| meiotic_recombination/meiotic_recombination_DSB_map_hg37.bed      | 907 KB   | `5c2f2a0dc60c23ab0cec2677fc1760ff` | TODO                 | TODO                                                                                                                             |
| **Histone Modifications**                                         |          |                                    |                      |                                                                                                                                  |
| histone_marks/ENCFF498ERO_HEK_H3K4me3_R1_hg38.bed                 | 2.04 MB  | `4a3d328e346e371d94e11092e030682b` | ENCODE               | ENCFF498ERO                                                                                                                      |
| histone_marks/ENCFF127YXW_HEK_H3K4me3_R2_hg38.bed                 | 2.06 MB  | `3630a67e70b4eaf38374be6182a47a23` | ENCODE               | ENCFF127YXW                                                                                                                      |
| histone_marks/ENCFF224LEB_HEK_H3K36me3_R1_2_hg38.bed              | 1.92 MB  | `dcf6131c8f6c072fe5985cfcdcba43b1` | ENCODE               | ENCFF224LEB                                                                                                                      |
| histone_marks/HEK293_H3K4me3_PRDM9-B_HA_transfected.bed           | 9.66 MB  | `2428a16f8171bbb191fa3fe56500838f` | GEO                  | GSE99407                                                                                                                         |
| histone_marks/HEK293_H3K4me3_untransfected.bed                    | 2.52 MB  | `abead5744038f61e51302e125e0ab913` | GEO                  | GSE99407                                                                                                                         |
| histone_marks/HEK293_H3K36me3_PRDM9-B-HA_transfected.bed          | 387 KB   | `065523e7dc1199cddae76235c32970ee` | GEO                  | GSE99407                                                                                                                         |
| histone_marks/HEK293_H3K36me3_untransfected.bed                   | 10.06 MB | `679d68ee97f1c1ffdb9288050c8a0a07` | GEO                  | GSE99407                                                                                                                         |


## Analysis Pipeline

### Stage 1: Window Sampling
- Derive PRDM9 windows from MACS2 peaks
- Generate matched (GC+entropy) and global (length-only) backgrounds

### Stage 2: Feature Generation (6 feature families)
1. G4 (pqsfinder + G4-seq support)
2. DNA methylation (RRBS)
3. Sequence motifs (PWM scanning)
4. Chromatin accessibility (ATAC-seq)
5. Recombination hotspots (DSB maps)
6. Histone modifications (7 ChIP-seq tracks)

### Stage 3: Enrichment Analysis
- Welch's t-test + block-wise Monte Carlo (200k iterations)
- BH FDR correction
- Final significance: p_final = min(p_welch, p_BG_null)

## Execution

### Local Development
```bash
mamba activate prdm9-analysis
jupyter lab
```

### Remote Cluster (Production)
```bash
# Connect
./scripts/ssh/connect.sh

# Execute notebook
./scripts/ssh/remote-exec.sh exec "cd /home/jovyan/prdm9_new && jupyter nbconvert --execute 1_sample_windows.ipynb --to notebook --inplace"

# Generate checksums
./scripts/ssh/remote-exec.sh exec "python scripts/cluster/generate_input_checksums.py"

# Pull results
./scripts/ssh/remote-exec.sh pull /home/jovyan/prdm9_new/input_checksums.json ./data_sources/

# Disconnect
./scripts/ssh/disconnect.sh
```

## File Integrity

Verify input files match expected checksums:

```bash
# On cluster
cd /home/jovyan/prdm9_new
python scripts/cluster/generate_input_checksums.py

# Compare with stored checksums
diff input_checksums.json ../data_sources/input_checksums.json
```

## Software Requirements

- Python 3.11+
- pandas, NumPy, SciPy, scikit-learn
- PyRanges, pyfaidx, pysam, pyBigWig
- matplotlib, joblib, tqdm, Rich
- Jupyter notebooks

Computationally intensive stages use up to 64 parallel workers.

## Data Availability

| Data Type | Source | Accession Numbers |
|-----------|--------|-------------------|
| PRDM9 ChIP-seq | GEO | GSE99407 |
| H3K4me3/H3K36me3 ChIP-seq | GEO | GSE99407 |
| BG4 CUT&Tag | SRA | SRR14300945, SRR14879748, SRR26244907, SRR26244908, SRR26244909, SRR26244910, SRR26244911, SRR26244912, SRR25010684, SRR25010685 |
| RRBS DNA methylation | ENCODE | ENCFF001TMQ, ENCFF001TMR |
| ATAC-seq | ENCODE | ENCSR205FUM, ENCFF668ATM |
| Histone marks (ENCODE) | ENCODE | ENCFF498ERO, ENCFF127YXW, ENCFF224LEB |

## Code Availability

- Main pipeline: https://github.com/tommsko/PRDM9-enrichment
- Additional code: https://github.com/angelika [TBD]

## Citation

Hartge MK, Pfennig Hernández N, Pavlík T, et al. The meiosis specific histone methyltransferase PRDM9 uses G-quadruplex structures as binding platforms. [Journal] [Year].

## Contact

**Corresponding:** Angelika Lahnsteiner (angelika.lahnsteiner@plus.ac.at)  
**Computational:** Tomáš Pavlík (Masaryk University, Brno)
