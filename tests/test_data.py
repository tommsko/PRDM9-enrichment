"""Input layer (prdm9_pipeline/data.py): manifest, transformation steps, verification."""
from __future__ import annotations

import gzip
import re
import subprocess
import sys

import pytest

from conftest import REPO
from prdm9_pipeline import data as D

MANIFEST = REPO / "config" / "inputs.yaml"


# --------------------------------------------------------------------------- manifest
def stage_input_paths():
    """Input files the stage scripts open, as quoted literals (e.g. "g4/pqsfinder_hg38.bed")."""
    dirs = "|".join(["prdm9_peaks", "g4", "methylation", "motifs", "accessibility", "meiotic_recombination", "histone_marks"])
    found = set()
    for f in (REPO / "prdm9_pipeline" / "stages").glob("s*.py"):
        found |= set(re.findall(rf"[\"'](hg38\.fa(?:\.fai)?|(?:{dirs})/[^\"'{{}}]+\.\w+)[\"']", f.read_text()))
    return found


def test_manifest_matches_stage_inputs_exactly():
    """Every input a stage opens is in the manifest, and the manifest holds nothing the stages do not use."""
    manifest = {e.path for e in D.load_manifest(MANIFEST)}
    used = stage_input_paths()
    assert used - manifest == set(), "inputs used by the stages but missing from config/inputs.yaml"
    assert manifest - used == set(), "manifest entries not used by any stage"


def test_manifest_entries():
    for e in D.load_manifest(MANIFEST):
        if e.kind == "derived":
            assert e.rule == "genome_index", e.rule      # only byte-identical rebuilds are allowed
        for step in e.steps:
            assert step in ("gunzip", "strip_track") or step.startswith("head:")


def test_manifest_rules_exist_in_workflow():
    text = (REPO / "workflow" / "rules" / "data.smk").read_text()
    for rule in {e.rule for e in D.load_manifest(MANIFEST) if e.rule}:
        assert f"rule {rule}:" in text


def test_manifest_rejects_bad_entries(tmp_path):
    bad = tmp_path / "m.yaml"
    bad.write_text("g:\n  a.bed: {kind: download}\n")
    with pytest.raises(ValueError, match="needs the md5"):
        D.load_manifest(bad)
    bad.write_text("g:\n  a.bed: {kind: download, md5: '%s', only_if: x}\n" % ("0" * 32))
    with pytest.raises(ValueError, match="unknown field"):
        D.load_manifest(bad)
    bad.write_text("g:\n  a.bed: {kind: download, md5: '%s'}\n" % ("0" * 32))
    with pytest.raises(ValueError, match="without url"):
        D.load_manifest(bad)
    bad.write_text("g:\n  a.bed: {kind: download, url: 'x', md5: 'abc'}\n")
    with pytest.raises(ValueError, match="malformed md5"):
        D.load_manifest(bad)
    bad.write_text("g:\n  a.bed: {kind: provided, md5: '%s'}\nh:\n  a.bed: {kind: provided, md5: '%s'}\n" % ("0" * 32, "0" * 32))
    with pytest.raises(ValueError, match="duplicate"):
        D.load_manifest(bad)


# --------------------------------------------------------------------------- primitives
def test_apply_steps(tmp_path):
    src = tmp_path / "x.gz"
    with gzip.open(src, "wt") as fh:
        fh.write('track name="RRBS" visibility=2\n' + "".join(f"chr1\t{i}\t{i + 1}\n" for i in range(10)))
    out = tmp_path / "x.bed"
    D.apply_steps(src, out, ["gunzip", "strip_track", "head:3"])
    assert out.read_text() == "chr1\t0\t1\nchr1\t1\t2\nchr1\t2\t3\n"
    # strip_track leaves files without a track line untouched; no steps = copy
    D.apply_steps(out, tmp_path / "y.bed", ["strip_track"])
    D.apply_steps(out, tmp_path / "z.bed", [])
    assert (tmp_path / "y.bed").read_bytes() == (tmp_path / "z.bed").read_bytes() == out.read_bytes()
    with pytest.raises(ValueError):
        D.apply_steps(out, tmp_path / "w", ["untar"])


def test_fetch_entry_file_url(tmp_path):
    src = tmp_path / "remote" / "f.txt.gz"
    src.parent.mkdir()
    with gzip.open(src, "wb") as fh:
        fh.write(b"line1\nline2\nline3\n")
    expect = tmp_path / "expect"
    expect.write_bytes(b"line1\nline2\n")
    e = D.Entry(path="sub/f.txt", kind="download", group="g", url=src.as_uri(), url_md5=D.md5sum(src),
                steps=["gunzip", "head:2"], md5=D.md5sum(expect))
    data = tmp_path / "data"
    assert D.fetch_entry(e, data, data / ".downloads").startswith("fetched")
    assert (data / "sub/f.txt").read_bytes() == b"line1\nline2\n"
    assert not list((data / ".downloads").iterdir())          # transformed downloads are not kept
    assert D.fetch_entry(e, data, data / ".downloads").startswith("present")
    e.url_md5 = "0" * 32
    (data / "sub/f.txt").unlink()
    with pytest.raises(RuntimeError, match="publisher checksum"):
        D.fetch_entry(e, data, data / ".downloads")


def test_verify(tmp_path):
    (tmp_path / "a").write_text("A")
    (tmp_path / "b").write_text("B")
    entries = [D.Entry(path="a", kind="download", group="g", url="u", md5=D.md5sum(tmp_path / "a")),
               D.Entry(path="b", kind="download", group="g", url="u", md5="0" * 32),
               D.Entry(path="c", kind="provided", group="g", md5="0" * 32)]
    report = tmp_path / "report.txt"
    assert D.verify(entries, tmp_path, report) == 2
    assert report.read_text().splitlines() == ["OK        a", f"MISMATCH  b  expected {'0' * 32} got {D.md5sum(tmp_path / 'b')}",
                                               "MISSING   c"]
    assert D.verify(entries[:1], tmp_path, report) == 0


def test_cli_provided_reports_missing(tmp_path, capsys):
    rc = D.main(["provided", "--manifest", str(MANIFEST), "--data", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    for path in ("g4/G4s_short_pqsfinder_hg38.txt", "methylation/HEK_RRBS_DNAmet_R1+R2_hg38.sorted.bed",
                 "meiotic_recombination/meiotic_recombination_DSB_map_hg38.bed"):
        assert path in err


def test_asset_name():
    assert D.asset_name("methylation/HEK_RRBS_DNAmet_R1+R2_hg38.sorted.bed") == "HEK_RRBS_DNAmet_R1.R2_hg38.sorted.bed"
    assert D.asset_name("histone_marks/HEK293_H3K36me3_PRDM9-B-HA_transfected.bed") == "HEK293_H3K36me3_PRDM9-B-HA_transfected.bed"


def test_provided_asset_names_are_unique():
    names = [D.asset_name(e.path) for e in D.load_manifest(MANIFEST) if e.kind == "provided"]
    assert len(names) == len(set(names)) == 9


def test_fetch_provided(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "x.y.bed").write_text("chr1\t1\t2\n")
    e = D.Entry(path="g/x+y.bed", kind="provided", group="g", md5=D.md5sum(remote / "x.y.bed"))
    assert D.fetch_provided(e, tmp_path / "data", remote.as_uri()).startswith("fetched")
    assert D.fetch_provided(e, tmp_path / "data", "").startswith("present")
    with pytest.raises(FileNotFoundError):
        D.fetch_provided(D.Entry(path="g/y.bed", kind="provided", group="g", md5="0" * 32), tmp_path / "data", "")


# --------------------------------------------------------------------------- derived files / workflow
def test_genome_index(tiny_genome, tmp_path):
    fa, seqs = tiny_genome
    D.genome_index(fa)
    fai = [l.split("\t") for l in (fa.parent / "tiny.fa.fai").read_text().splitlines()]
    assert [(r[0], int(r[1])) for r in fai] == [(k, len(v)) for k, v in seqs.items()]


def test_snakemake_dry_run(tmp_path):
    """The workflow parses and its DAG holds exactly the data rules and the 11 manuscript stages."""
    smk = D.Path(sys.executable).parent / "snakemake"
    if not smk.exists():
        pytest.skip("snakemake not installed")
    cmd = [str(smk), "all", "-s", str(REPO / "workflow" / "Snakefile"), "--directory", str(REPO), "-n", "--quiet", "rules",
           "--cores", "4", "--config", f"data_dir={tmp_path}/data", f"results_dir={tmp_path}/results"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    jobs = {line.split()[0] for line in res.stdout.splitlines() if re.match(r"^\w+\s+\d+$", line)} - {"total"}
    stages = {p.stem for p in (REPO / "prdm9_pipeline" / "stages").glob("s*.py")}
    assert jobs == stages | {"all", "workspace", "data", "downloads", "provided", "genome_index"}, jobs
