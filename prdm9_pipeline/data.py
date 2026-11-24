"""Input data: download and verify (driven by config/inputs.yaml).

Used by the data rules of the workflow (workflow/rules/data.smk) as

    python -m prdm9_pipeline.data downloads   --manifest config/inputs.yaml --data data [--threads N]
    python -m prdm9_pipeline.data provided    --manifest ... --data ... --url <provided_inputs_url>
    python -m prdm9_pipeline.data verify      --manifest ... --data ...

Every input is the exact file used for the manuscript (md5-checked); nothing is re-derived differently.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "prdm9-enrichment-pipeline (+https://github.com/tommsko/PRDM9-enrichment)"


# --------------------------------------------------------------------------- manifest
@dataclass
class Entry:
    path: str                      # relative to the data directory
    kind: str                      # download | derived | provided
    group: str
    md5: str                       # md5 of the file exactly as used
    url: str | None = None
    url_md5: str | None = None
    steps: list = field(default_factory=list)
    rule: str | None = None
    source: str = ""


def load_manifest(path: Path) -> list[Entry]:
    """Flatten config/inputs.yaml into entries and validate them."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    entries = []
    for group, items in raw.items():
        for rel, spec in items.items():
            spec = dict(spec)
            text = (spec.pop("source", "") or spec.pop("recipe", "") or "").strip()
            spec.pop("recipe", None)
            unknown = set(spec) - set(Entry.__dataclass_fields__)
            if unknown:
                raise ValueError(f"{rel}: unknown field(s) {sorted(unknown)}")
            if not spec.get("md5"):
                raise ValueError(f"{rel}: every input needs the md5 of the file used")
            entries.append(Entry(path=rel, group=group, source=text, **spec))
    seen = set()
    for e in entries:
        if e.path in seen:
            raise ValueError(f"duplicate manifest path {e.path}")
        seen.add(e.path)
        if e.kind not in ("download", "derived", "provided"):
            raise ValueError(f"{e.path}: unknown kind {e.kind}")
        if e.kind == "download" and not e.url:
            raise ValueError(f"{e.path}: download without url")
        if e.kind == "derived" and not e.rule:
            raise ValueError(f"{e.path}: derived without rule")
        for h in (e.md5, e.url_md5):
            if h is not None and (len(h) != 32 or any(c not in "0123456789abcdef" for c in h)):
                raise ValueError(f"{e.path}: malformed md5 {h!r}")
    return entries


# --------------------------------------------------------------------------- primitives
def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> Path:
    """Resumable HTTP(S) download to `dest` (via dest.part); returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with open(part, mode) as out:
                    shutil.copyfileobj(resp, out, 1 << 22)
            part.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - network errors of all kinds are retried
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}: {exc}") from exc
            time.sleep(min(60, 5 * attempt))
    return dest


def apply_steps(src: Path, dest: Path, steps: list) -> None:
    """Transform a downloaded file into the input file. Steps: gunzip, strip_track, head:N."""
    current = src
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        for i, step in enumerate(steps):
            out = Path(tmp) / f"step{i}"
            if step == "gunzip":
                with gzip.open(current, "rb") as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 22)
            elif step == "strip_track":
                with open(current, "rb") as fi, open(out, "wb") as fo:
                    first = fi.readline()
                    if not first.startswith(b"track"):
                        fo.write(first)
                    shutil.copyfileobj(fi, fo, 1 << 22)
            elif isinstance(step, str) and step.startswith("head:"):
                n = int(step.split(":", 1)[1])
                with open(current, "rb") as fi, open(out, "wb") as fo:
                    for _, line in zip(range(n), fi):
                        fo.write(line)
            else:
                raise ValueError(f"unknown step {step!r}")
            current = out
        if current == src:
            shutil.copyfile(src, dest)
        else:
            shutil.move(str(current), dest)


def fetch_entry(e: Entry, data: Path, raw_dir: Path) -> str:
    """Download + transform one entry unless the final file already has the expected md5."""
    final = data / e.path
    if final.exists() and md5sum(final) == e.md5:
        return f"present  {e.path}"
    raw = raw_dir / (e.path.replace("/", "__") + ".download")
    if not raw.exists() or (e.url_md5 and md5sum(raw) != e.url_md5):
        download(e.url, raw)
    if e.url_md5 and md5sum(raw) != e.url_md5:
        raw.unlink()
        raise RuntimeError(f"{e.path}: downloaded object does not match the publisher checksum {e.url_md5}")
    final.parent.mkdir(parents=True, exist_ok=True)
    apply_steps(raw, final, e.steps)
    if e.steps:
        raw.unlink()
    return f"fetched  {e.path}"


def fetch_provided(e: Entry, data: Path, url: str) -> str:
    """Fetch a `provided` input from <url>/<file name>; returns a status line, or raises if unavailable."""
    final = data / e.path
    if final.exists() and md5sum(final) == e.md5:
        return f"present  {e.path}"
    if not url:
        raise FileNotFoundError(e.path)
    final.parent.mkdir(parents=True, exist_ok=True)
    download(f"{url.rstrip('/')}/{Path(e.path).name}", final)
    if md5sum(final) != e.md5:
        raise RuntimeError(f"{e.path}: md5 mismatch after download from provided_inputs_url")
    return f"fetched  {e.path}"


# --------------------------------------------------------------------------- derived files
def genome_index(fasta: Path) -> None:
    import pysam

    pysam.faidx(str(fasta))


# --------------------------------------------------------------------------- verification
def verify(entries: list[Entry], data: Path, report: Path) -> int:
    """md5 every input against the file used; write a report; return the number of problems."""
    errors, lines = 0, []
    for e in entries:
        f = data / e.path
        if not f.exists():
            errors += 1
            lines.append(f"MISSING   {e.path}")
            continue
        got = md5sum(f)
        if got == e.md5:
            lines.append(f"OK        {e.path}")
        else:
            errors += 1
            lines.append(f"MISMATCH  {e.path}  expected {e.md5} got {got}")
    report.write_text("\n".join(lines) + "\n")
    return errors


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["downloads", "provided", "verify"])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--url", default="", help="provided_inputs_url")
    args = ap.parse_args(argv)
    entries = load_manifest(args.manifest)
    data = args.data
    data.mkdir(parents=True, exist_ok=True)

    if args.action == "downloads":
        todo = [e for e in entries if e.kind == "download"]
        raw = data / ".downloads"
        with cf.ThreadPoolExecutor(max_workers=max(1, min(args.threads, 6))) as ex:
            for msg in ex.map(lambda e: fetch_entry(e, data, raw), todo):
                print(msg, flush=True)
        return 0

    if args.action == "provided":
        missing = []
        for e in (x for x in entries if x.kind == "provided"):
            try:
                print(fetch_provided(e, data, args.url), flush=True)
            except FileNotFoundError:
                missing.append(e.path)
        if missing:
            print("These inputs have no public source and must be supplied (set provided_inputs_url in "
                  "config/config.yaml, or copy them into the data directory):\n  " + "\n  ".join(missing),
                  file=sys.stderr)
            return 2
        return 0

    errors = verify(entries, data, data / "verification_report.txt")
    print((data / "verification_report.txt").read_text())
    if errors:
        print(f"{errors} input(s) missing or not matching the manuscript inputs", file=sys.stderr)
        return 1
    (data / ".verified").write_text(dt.datetime.now().isoformat(timespec="seconds") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
