# PRDM9 Enrichment Analysis

**The meiosis specific histone methyltransferase PRDM9 uses G-quadruplex structures as binding platforms**
**Preprint available at: TBD**

Computational pipeline characterising PRDM9 binding sites in HEK293 cells: PRDM9 ChIP-seq windows against
matched and genome-wide backgrounds, scored for G-quadruplexes, DNA methylation, PRDM9 motifs, chromatin
accessibility, meiotic DSB hotspots and histone marks. For the methodology, see the article.

## Quick start

Requires Linux, Python ≥ 3.8, and one of mamba, micromamba or conda (micromamba can be bootstrapped).

```bash
git clone https://github.com/tommsko/PRDM9-enrichment
cd PRDM9-enrichment

python prdm9.py doctor     # check machine, tools, environment, network
python prdm9.py setup      # create the pinned environment in .envs/ (add --bootstrap if no conda/mamba)
python prdm9.py fetch      # download the input data and verify it against the files used (md5)
python prdm9.py run        # run the manuscript notebooks, in order
```

`prdm9.py` uses only the Python standard library; everything else lives in the environment it creates.

| Command | What it does |
|---|---|
| `doctor [--offline]` | checks the machine and setup: <ul><li>OS, CPUs</li><li>RAM: ≥ 64 GB, 256 GB recommended</li><li>free disk: ~30 GB</li><li>conda manager, environment and its packages</li><li>data hosts reachable</li><li>inputs verified</li></ul> |
| `setup [--update] [--bootstrap]` | creates `.envs/prdm9` from `envs/prdm9.yml` |
| `fetch` | downloads every input listed in `config/inputs.yaml` and writes `data/verification_report.txt` |
| `run [target ...] [--cores N] [--dry-run] [--config key=value ...]` | runs the workflow (default target `all`) |
| `status` | completed steps and pending jobs |
| `test [pytest args]` | unit tests (synthetic data; no download needed) |

Targets:

- `all`: all 11 notebook stages (1 → 2a–2f → 2g → 3a–3c).
- `fig2a`: only the steps needed for Fig 2A (stage 3c).
- `fig3`: only the steps needed for Fig 3 (stage 3a).
- `data`: same as `fetch`.

The workflow resumes where it stopped. Each stage logs to `results/logs/<stage>.log` and is marked done in
`results/.done/<stage>`.

## Configuration

Settings live in `config/config.yaml`. Edit the file to change a setting permanently, or override it for a
single run, e.g. `python prdm9.py run --config threads=32`.

| Key | Default | Meaning |
|---|---|---|
| `threads` | 64 | worker processes per stage |
| `mc_iterations` / `bootstrap` | 200000 / 5000 | Monte Carlo iterations (3a, 3c) and bootstrap replicates (3b); lower them only for smoke runs |
| `provided_inputs_url` | empty | where to fetch the inputs that have no public source (see below) |

The defaults are the manuscript settings.

## Inputs

`config/inputs.yaml` lists every input the notebooks read. For each one it gives the source URL, any format
step (gunzip, dropping a track line, taking the first N lines), and the md5 of the file used for the
manuscript. `fetch` fails if any file differs.

- **Downloaded from the original source:**
  - hg38 (UCSC, initial release);
  - the pqsfinder hg38 track;
  - from the authors' GitHub at a pinned commit: the PRDM9 peaks and the BG4 CUT&Tag consensus;
  - ENCODE ATAC and histone peaks;
  - Altemose 2017 motifs.
- **Rebuilt:** only the genome index (`hg38.fa.fai`), which comes out byte-identical.
- **Provided:** these files were made by the authors with undocumented steps and are not published anywhere.
  They are fetched from `provided_inputs_url` (or copied into `data/` by hand) and md5-checked:
  - the pqsfinder loop-class files;
  - four GSE99407 histone tracks;
  - the merged RRBS file;
  - the hg38 DSB map.

## Outputs

The workflow computes inside `results/` (config `results_dir`). That directory links to the inputs in `data/`,
mirroring the directory the notebooks were developed in.

| Stage | Output folder in `results/` |
|---|---|
| 1 sample windows | `data_prdm9_sampling_production/` |
| 2a–2f features | `data_prdm9_{g4,methylation,motifs,accessibility,recombination,histones}_production/` |
| 2g combine | `data_prdm9_combined_production/` |
| 3a G4 enrichment (Fig 3) | `data_prdm9_g4_enrichment_main_figure_production/` |
| 3b G4-seq support | `data_prdm9_g4seq_support_normalized_publication_v2/` |
| 3c all features (Fig 2A) | `data_prdm9_all_feature_enrichment_production/` |

## Repository layout

```
prdm9.py                  command-line entry point (doctor / setup / fetch / run / status / test)
config/config.yaml        run settings
config/inputs.yaml        input manifest: sources, format steps, checksums
envs/prdm9.yml            pinned conda environment
workflow/Snakefile        the notebook stages and targets
workflow/rules/data.smk   download / verify inputs
notebooks/                the manuscript notebooks (source of the stage scripts)
prdm9_pipeline/stages/    notebooks converted to scripts (generated: python tools/convert_notebooks.py)
prdm9_pipeline/data.py    input download and verification
tests/                    unit tests
```

The stage scripts are generated from the notebooks; their analysis code is unchanged. After editing a notebook,
regenerate them with `python tools/convert_notebooks.py`. `python tools/convert_notebooks.py --check` and the
unit tests catch stale scripts.

The converter only makes mechanical edits:

- drops `pip install` lines;
- prints `display()` output to the log;
- makes thread counts, Monte Carlo iterations and bootstrap replicates configurable, defaulting to the notebook
  values.

## Citation

Hartge MK, Pfennig Hernández N, Pavlík T, et al. The meiosis specific histone methyltransferase PRDM9 uses
G-quadruplex structures as binding platforms. [Journal] [Year].

## Contact

**Corresponding:** Angelika Lahnsteiner (angelika.lahnsteiner@plus.ac.at)
**Computational:** Tomáš Pavlík (pavlik@ics.muni.cz)
