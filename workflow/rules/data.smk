# Inputs: everything under data/ is the exact file used for the manuscript, downloaded
# from its original source (manifest and provenance: config/inputs.yaml; code: prdm9_pipeline/data.py).
#
#   downloads      every `download` entry (parallel, resumable, publisher checksums)
#   provided       inputs without a public source for the exact file (config provided_inputs_url)
#   genome_index   hg38.fa.fai (byte-identical to the original index)
#   data           md5 of every input vs the files used -> data/.verified + data/verification_report.txt

import sys

sys.path.insert(0, str(REPO))
from prdm9_pipeline import data as D

MANIFEST = REPO / "config" / "inputs.yaml"
STAMPS = DATA / ".stamps"


def data_cli(action, extra=""):
    """Shell command for prdm9_pipeline.data."""
    return f"env PYTHONPATH={REPO} python -m prdm9_pipeline.data {action} --manifest {MANIFEST} --data {DATA} {extra}"


rule downloads:
    output: touch(STAMPS / "downloads")
    threads: 6
    log: DATA / "logs" / "downloads.log"
    shell: data_cli("downloads", "--threads {threads}") + " 2>&1 | tee {log}"


rule provided:
    output: touch(STAMPS / "provided")
    params: url=config.get("provided_inputs_url", "") or ""
    log: DATA / "logs" / "provided.log"
    shell: data_cli("provided", "--url '{params.url}'") + " 2>&1 | tee {log}"


rule genome_index:
    input: STAMPS / "downloads"
    output: DATA / "hg38.fa.fai"
    run: D.genome_index(DATA / "hg38.fa")


rule data:
    """All inputs present and identical to the files used."""
    input:
        STAMPS / "downloads",
        STAMPS / "provided",
        rules.genome_index.output,
    output: DATA / ".verified"
    shell: data_cli("verify")
