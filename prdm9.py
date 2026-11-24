#!/usr/bin/env python3
"""PRDM9 enrichment pipeline: one entry point for setting up and running the analysis.

    python prdm9.py doctor            check the machine (OS, CPUs, RAM, disk, tools, network, envs)
    python prdm9.py setup             create the pinned conda environment in .envs/
    python prdm9.py fetch             download + verify all input data into data/
    python prdm9.py run [TARGET ...]  run the workflow (default target: all)
    python prdm9.py status            what has been computed, where the results are
    python prdm9.py test              run the unit tests

Targets: all (manuscript notebooks 1-3c), data, fig2a, fig3, or any rule name from workflow/Snakefile.
Options for `run`: --cores N, --dry-run, --config key=value ... (see config/config.yaml).

This file uses only the Python standard library so that it works before any environment exists.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
ENV_DIR = REPO / ".envs"
TOOLS_DIR = REPO / ".tools"
ENVS = {"prdm9": REPO / "envs" / "prdm9.yml"}
SNAKEFILE = REPO / "workflow" / "Snakefile"
CONFIG = REPO / "config" / "config.yaml"
MICROMAMBA_URL = {"Linux-x86_64": "https://micro.mamba.pm/api/micromamba/linux-64/latest",
                  "Linux-aarch64": "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"}
REQUIRED_IMPORTS = ["numpy", "pandas", "scipy", "matplotlib", "sklearn", "joblib", "numba", "pyarrow", "tqdm", "rich",
                    "pyranges", "pyfaidx", "pysam", "pyBigWig", "snakemake", "yaml", "pytest"]
DATA_HOSTS = ["hgdownload.soe.ucsc.edu", "www.encodeproject.org", "ftp.ncbi.nlm.nih.gov",
              "raw.githubusercontent.com", "pqsfinder.fi.muni.cz", "cdn.elifesciences.org"]
MIN_RAM_GB, REC_RAM_GB = 64, 256
DISK_GB = {"data": 15, "results": 15}

GREEN, YELLOW, RED, RESET = ("\033[32m", "\033[33m", "\033[31m", "\033[0m") if sys.stdout.isatty() else ("",) * 4


def say(status: str, msg: str, hint: str = "") -> None:
    mark = {"ok": f"{GREEN}✓{RESET}", "warn": f"{YELLOW}!{RESET}", "fail": f"{RED}✗{RESET}", "info": "·"}[status]
    print(f" {mark} {msg}" + (f"\n     → {hint}" if hint else ""))


# --------------------------------------------------------------------------- helpers
def find_manager() -> str | None:
    """Prefer mamba > micromamba > conda; a bootstrapped micromamba in .tools/ counts too."""
    for name in ("mamba", "micromamba", "conda"):
        path = shutil.which(name)
        if path:
            return path
    local = TOOLS_DIR / "bin" / "micromamba"
    return str(local) if local.exists() else None


def env_prefix(name: str) -> Path:
    return ENV_DIR / name


def env_bin(name: str, tool: str) -> Path:
    return env_prefix(name) / "bin" / tool


def mem_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        return None
    return None


def free_gb(path: Path) -> float:
    path = path if path.exists() else path.parent
    return shutil.disk_usage(path).free / 1024 ** 3


def read_config_value(key: str, default: str) -> str:
    """Minimal reader for top-level `key: value` lines of config/config.yaml (no YAML dependency)."""
    if CONFIG.exists():
        for line in CONFIG.read_text().splitlines():
            if line.split("#")[0].strip().startswith(f"{key}:"):
                return line.split("#")[0].split(":", 1)[1].strip().strip("'\"") or default
    return default


def reachable(host: str, timeout: float = 5.0) -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    problems = 0
    print("System")
    system = f"{platform.system()}-{platform.machine()}"
    if platform.system() == "Linux":
        say("ok", f"OS: {system}")
    else:
        problems += 1
        say("fail", f"OS: {system}", "the pipeline needs Linux (fork-based multiprocessing, bioconda tools); use a Linux server/cluster")
    py = sys.version_info
    say("ok" if py >= (3, 8) else "fail", f"Python for this launcher: {platform.python_version()}",
        "" if py >= (3, 8) else "use Python >= 3.8 to run prdm9.py (the analysis env brings its own Python 3.11)")
    cpus = os.cpu_count() or 1
    threads = int(read_config_value("threads", "64"))
    say("ok" if cpus >= 16 else "warn", f"CPUs: {cpus} (config threads: {threads})",
        "" if cpus >= 16 else "runs will be slow; the manuscript runs used 64 workers")
    ram = mem_gb()
    if ram is None:
        say("warn", "RAM: unknown")
    else:
        status = "ok" if ram >= REC_RAM_GB else ("warn" if ram >= MIN_RAM_GB else "fail")
        problems += status == "fail"
        say(status, f"RAM: {ram:.0f} GB", "" if status == "ok" else f"recommended >= {REC_RAM_GB} GB (minimum ~{MIN_RAM_GB} GB)")
    data_dir = REPO / read_config_value("data_dir", "data")
    results_dir = REPO / read_config_value("results_dir", "results")
    need = DISK_GB["data"] + DISK_GB["results"]
    free = free_gb(data_dir)
    say("ok" if free >= need else "fail", f"Free disk at {data_dir.parent}: {free:.0f} GB (need ~{need} GB)")
    problems += free < need
    say("ok" if shutil.which("git") else "warn", "git: " + (shutil.which("git") or "not found"))

    print("Package managers")
    mgr = find_manager()
    if mgr:
        say("ok", f"conda-family manager: {mgr}")
    else:
        say("fail", "no mamba / micromamba / conda found", "run `python prdm9.py setup --bootstrap` to download a local micromamba")
        problems += 1
    say("ok" if shutil.which("pip") or shutil.which("pip3") else "info", "pip: " + str(shutil.which("pip") or shutil.which("pip3")))

    print("Environments")
    py_path = env_bin("prdm9", "python")
    if not py_path.exists():
        say("fail", f"env prdm9: missing ({env_prefix('prdm9')})", "run `python prdm9.py setup`")
        problems += 1
    else:
        code = "import importlib,sys; m=[x for x in sys.argv[1:] if importlib.util.find_spec(x) is None]; print(','.join(m))"
        out = subprocess.run([str(py_path), "-c", code, *REQUIRED_IMPORTS], capture_output=True, text=True)
        missing = out.stdout.strip()
        say("ok" if not missing else "fail", f"env prdm9: {'all packages import' if not missing else 'missing ' + missing}",
            "" if not missing else "run `python prdm9.py setup --update`")
        problems += bool(missing)

    if not args.offline:
        print("Network (data sources)")
        for host in DATA_HOSTS:
            ok = reachable(host)
            say("ok" if ok else "warn", f"{host}", "" if ok else "unreachable: `fetch` will fail for files from this host")

    print("Data")
    stamp = data_dir / ".verified"
    if stamp.exists():
        say("ok", f"inputs verified ({stamp.read_text().strip()})")
    else:
        say("info", f"inputs not fetched/verified yet in {data_dir}", "run `python prdm9.py fetch`")
    print()
    print("All checks passed." if problems == 0 else f"{problems} problem(s) found.")
    return 1 if problems else 0


# --------------------------------------------------------------------------- setup
def bootstrap_micromamba() -> str:
    key = f"{platform.system()}-{platform.machine()}"
    if key not in MICROMAMBA_URL:
        sys.exit(f"no micromamba bootstrap for {key}; install mamba/conda manually")
    TOOLS_DIR.mkdir(exist_ok=True)
    archive = TOOLS_DIR / "micromamba.tar.bz2"
    print(f"downloading micromamba from {MICROMAMBA_URL[key]} ...")
    urllib.request.urlretrieve(MICROMAMBA_URL[key], archive)
    with tarfile.open(archive) as tar:
        member = tar.getmember("bin/micromamba")
        tar.extractall(TOOLS_DIR, members=[member])
    archive.unlink()
    path = TOOLS_DIR / "bin" / "micromamba"
    path.chmod(0o755)
    return str(path)


def create_env(manager: str, name: str, update: bool) -> None:
    prefix = env_prefix(name)
    spec = ENVS[name]
    is_micro = Path(manager).name == "micromamba"
    if prefix.exists() and not update:
        print(f"env {name} already exists at {prefix} (use --update to re-solve)")
        return
    if prefix.exists():
        cmd = [manager, "env", "update", "-p", str(prefix), "-f", str(spec), "--prune"]
        if is_micro:
            cmd = [manager, "update", "-y", "-p", str(prefix), "-f", str(spec)]
    else:
        cmd = [manager, "env", "create", "-p", str(prefix), "-f", str(spec)]
        if is_micro:
            cmd = [manager, "create", "-y", "-p", str(prefix), "-f", str(spec)]
    env = dict(os.environ)
    if is_micro:
        env.setdefault("MAMBA_ROOT_PREFIX", str(TOOLS_DIR / "mamba-root"))
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def cmd_setup(args) -> int:
    manager = find_manager()
    if manager is None or args.bootstrap:
        manager = bootstrap_micromamba() if (args.bootstrap or manager is None) else manager
    ENV_DIR.mkdir(exist_ok=True)
    create_env(manager, "prdm9", args.update)
    print("\nEnvironment ready. Next: python prdm9.py doctor && python prdm9.py fetch")
    return 0


# --------------------------------------------------------------------------- run / fetch / status / test
def snakemake(targets: list[str], cores: int | None, dry_run: bool, config: list[str], extra: list[str] | None = None) -> int:
    smk = env_bin("prdm9", "snakemake")
    if not smk.exists():
        sys.exit("environment missing: run `python prdm9.py setup` first")
    cores = cores or int(read_config_value("threads", str(os.cpu_count() or 1)))
    # targets first: --config takes all following words as key=value pairs
    cmd = [str(smk), *(targets or ["all"]), "-s", str(SNAKEFILE), "--directory", str(REPO), "--cores", str(cores),
           "--rerun-incomplete", "--printshellcmds"]
    if dry_run:
        cmd.append("--dry-run")
    if config:
        cmd += ["--config", *config]
    cmd += extra or []
    env = dict(os.environ, PATH=f"{env_prefix('prdm9') / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    print("+", " ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def cmd_run(args) -> int:
    return snakemake(args.targets, args.cores, args.dry_run, args.config)


def cmd_fetch(args) -> int:
    return snakemake(["data"], args.cores, args.dry_run, args.config)


def cmd_status(args) -> int:
    results = REPO / read_config_value("results_dir", "results")
    done = results / ".done"
    print(f"results directory: {results}")
    if not done.exists():
        print("nothing computed yet (run `python prdm9.py run`)")
        return 0
    for f in sorted(done.iterdir()):
        print(f"  ✓ {f.name}  ({f.read_text().strip()})")
    return snakemake(["all"], 1, True, [], ["--quiet", "rules"])


def cmd_test(args) -> int:
    py = env_bin("prdm9", "python")
    if not py.exists():
        sys.exit("environment missing: run `python prdm9.py setup` first")
    return subprocess.run([str(py), "-m", "pytest", str(REPO / "tests"), *args.pytest_args], cwd=REPO).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prdm9.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("doctor", help="check the machine and environments")
    p.add_argument("--offline", action="store_true", help="skip network checks")
    p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("setup", help="create the conda environment")
    p.add_argument("--update", action="store_true", help="re-solve existing environments")
    p.add_argument("--bootstrap", action="store_true", help="download a local micromamba into .tools/ and use it")
    p.set_defaults(func=cmd_setup)
    for name, func, hlp in [("fetch", cmd_fetch, "download, verify and rebuild input data"), ("run", cmd_run, "run the workflow")]:
        p = sub.add_parser(name, help=hlp)
        if name == "run":
            p.add_argument("targets", nargs="*", help="workflow targets (default: all)")
        p.add_argument("--cores", type=int, help="CPU cores for the workflow (default: config threads)")
        p.add_argument("--dry-run", "-n", action="store_true", help="only show what would run")
        p.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE", help="override config/config.yaml values")
        p.set_defaults(func=func)
    p = sub.add_parser("status", help="show what has been computed")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("test", help="run unit tests")
    p.add_argument("pytest_args", nargs="*", help="extra pytest arguments")
    p.set_defaults(func=cmd_test)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
