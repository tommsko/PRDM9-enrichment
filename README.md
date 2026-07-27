# PRDM9 Enrichment Analysis

**The meiosis specific histone methyltransferase PRDM9 uses G-​​quadruplex structures as binding platforms**
**Preprint available at: TBD**

Computational pipeline for characterizing PRDM9 binding characteristics in HEK293 cells. For methodology overview, please refer to the article.

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

| File                                                              | Size     | MD5 Checksum                       | Source                     | Accession                                                                                                                        |
| ----------------------------------------------------------------- | -------- | ---------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Reference Genome**                                              |          |                                    |                            |                                                                                                                                  |
| hg38.fa                                                           | 3.05 GB  | `b2aee9f885accc00531e59c4736bee63` | UCSC                       |                                                                                                                                  |
| **PRDM9 ChIP-seq**                                                |          |                                    |                            |                                                                                                                                  |
| prdm9_peaks/HEK293_PRDM9-B_HA_MACS2_hg38.bed                      | 1.44 MB  | `6dc00b4021b540eb0d504f45d54b4021` | GEO                        | GSE99407                                                                                                                         |
| **G4 Predictions (pqsfinder)**                                    |          |                                    |                            |                                                                                                                                  |
| g4/pqsfinder_hg38.bed                                             | 39.81 MB | `6bc2d5dceec34a9b593dd751a417c73a` | pqsfinder, Hon et al. 2017 |                                                                                                                                  |
| g4/G4s_short_pqsfinder_hg38.txt                                   | 34.67 MB | `b1dc6251a23d80285027470b78711ee3` | pqsfinder_hg38.bed subset  |                                                                                                                                  |
| g4/G4s_median_pqsfinder_hg38.txt                                  | 39.99 MB | `7f5736526cf39a1babfd6e3520ab1118` | pqsfinder_hg38.bed subset  |                                                                                                                                  |
| g4/G4s_long_pqsfinder_hg38.txt                                    | 33.69 MB | `d2d5ab988620aaafd4fab61fc4dc4a6f` | pqsfinder_hg38.bed subset  |                                                                                                                                  |
| **G4 Experimental (CUT&Tag)**                                     |          |                                    |                            |                                                                                                                                  |
| g4/HEK293_G4s_hg38.bed                                            | 1 MB     | `29655117b69960051bacb390c801c33c` | SRA                        | SRR14300945, SRR14879748, SRR26244907, SRR26244908, SRR26244909, SRR26244910, SRR26244911, SRR26244912, SRR25010684, SRR25010685 |
| **DNA Methylation (RRBS)**                                        |          |                                    |                            |                                                                                                                                  |
| methylation/ENCFF001TMQ_HEK_RRBS_DNAmet_R1_hg38.bed               | 109 MB   | `3441dfc226422f3790dd83c8a3497839` | ENCODE                     | ENCFF001TMQ                                                                                                                      |
| methylation/ENCFF001TMQ_HEK_RRBS_DNAmet_R1_hg38.without_track.bed | 109 MB   | `d9e4cc7aa5d03b5bce840cfcd2e80b4e` | ENCODE                     | ENCFF001TMQ                                                                                                                      |
| methylation/ENCFF001TMR_HEK_RRBS_DNAmet_R2_hg38.bed               | 93 MB    | `677357ae1325d6a821c5b8d6f0c505e2` | ENCODE                     | ENCFF001TMR                                                                                                                      |
| methylation/ENCFF001TMR_HEK_RRBS_DNAmet_R2_hg38.without_track.bed | 93 MB    | `ff369f69a140dc08b8a843a1381bbd84` | ENCODE                     | ENCFF001TMR                                                                                                                      |
| **Sequence Motifs**                                               |          |                                    |                            |                                                                                                                                  |
| motifs/motifs.txt                                                 | 37 KB    | `76d36747cb7bb6042dd01f21c22199b2` | Altemose et al. 2017       | Figure 1—source data 2. PWMs for all motifs, in MEME format (elife-28383-fig1-data2.txt)                                         |
| **Chromatin Accessibility (ATAC-seq)**                            |          |                                    |                            |                                                                                                                                  |
| accessibility/ENCFF668ATM.bigWig                                  | 612 MB   | `0c1426af95a49f253599359224667995` | ENCODE                     | ENCSR205FUM                                                                                                                      |
| **Meiotic Recombination Hotspots**                                |          |                                    |                            |                                                                                                                                  |
| meiotic_recombination/meiotic_recombination_DSB_map_hg38.bed      | 1.77 MB  | `b380b6d3382db7d966ce5cf6c93bca24` | GEO                        | GSE59836                                                                                                                         |
| **Histone Modifications**                                         |          |                                    |                            |                                                                                                                                  |
| histone_marks/ENCFF498ERO_HEK_H3K4me3_R1_hg38.bed                 | 2.04 MB  | `4a3d328e346e371d94e11092e030682b` | ENCODE                     | ENCFF498ERO                                                                                                                      |
| histone_marks/ENCFF127YXW_HEK_H3K4me3_R2_hg38.bed                 | 2.06 MB  | `3630a67e70b4eaf38374be6182a47a23` | ENCODE                     | ENCFF127YXW                                                                                                                      |
| histone_marks/ENCFF224LEB_HEK_H3K36me3_R1_2_hg38.bed              | 1.92 MB  | `dcf6131c8f6c072fe5985cfcdcba43b1` | ENCODE                     | ENCFF224LEB                                                                                                                      |
| histone_marks/HEK293_H3K4me3_PRDM9-B_HA_transfected.bed           | 9.66 MB  | `2428a16f8171bbb191fa3fe56500838f` | GEO                        | GSE99407                                                                                                                         |
| histone_marks/HEK293_H3K4me3_untransfected.bed                    | 2.52 MB  | `abead5744038f61e51302e125e0ab913` | GEO                        | GSE99407                                                                                                                         |
| histone_marks/HEK293_H3K36me3_PRDM9-B-HA_transfected.bed          | 387 KB   | `065523e7dc1199cddae76235c32970ee` | GEO                        | GSE99407                                                                                                                         |
| histone_marks/HEK293_H3K36me3_untransfected.bed                   | 10.06 MB | `679d68ee97f1c1ffdb9288050c8a0a07` | GEO                        | GSE99407                                                                                                                         |

Hartge MK, Pfennig Hernández N, Pavlík T, et al. The meiosis specific histone methyltransferase PRDM9 uses G-quadruplex structures as binding platforms. [Journal] [Year].

## Contact

**Corresponding:** Angelika Lahnsteiner (angelika.lahnsteiner@plus.ac.at)  
**Computational:** Tomáš Pavlík (pavlik@ics.muni.cz)
