"""prdm9.py: config reading, manager discovery, environment specs and command composition (nothing is run)."""
from __future__ import annotations

import importlib.util
import subprocess

import pytest
import yaml

from conftest import REPO

spec = importlib.util.spec_from_file_location("prdm9_cli", REPO / "prdm9.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_read_config_value_matches_yaml():
    cfg = yaml.safe_load(cli.CONFIG.read_text())
    for key in ("data_dir", "results_dir", "threads", "mc_iterations", "bootstrap"):
        assert cli.read_config_value(key, "?").lower() == str(cfg[key]).lower(), key
    assert cli.read_config_value("provided_inputs_url", "?").startswith("https://github.com/")
    assert cli.read_config_value("no_such_key", "x") == "x"


def test_find_manager_preference(monkeypatch, tmp_path):
    available = {"conda": "/opt/conda/bin/conda", "micromamba": "/usr/bin/micromamba"}
    monkeypatch.setattr(cli.shutil, "which", lambda name: available.get(name))
    assert cli.find_manager() == "/usr/bin/micromamba"
    available["mamba"] = "/opt/conda/bin/mamba"
    assert cli.find_manager() == "/opt/conda/bin/mamba"
    available.clear()
    monkeypatch.setattr(cli, "TOOLS_DIR", tmp_path)
    assert cli.find_manager() is None
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "micromamba").touch()
    assert cli.find_manager() == str(tmp_path / "bin" / "micromamba")


@pytest.mark.parametrize("manager,exists,update,expected", [
    ("/x/mamba", False, False, ["/x/mamba", "env", "create", "-p"]),
    ("/x/micromamba", False, False, ["/x/micromamba", "create", "-y", "-p"]),
    ("/x/conda", True, True, ["/x/conda", "env", "update", "-p"]),
    ("/x/micromamba", True, True, ["/x/micromamba", "update", "-y", "-p"]),
])
def test_create_env_commands(monkeypatch, tmp_path, manager, exists, update, expected):
    calls = []
    monkeypatch.setattr(cli, "ENV_DIR", tmp_path)
    if exists:
        (tmp_path / "prdm9").mkdir()
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    cli.create_env(manager, "prdm9", update)
    cmd, kw = calls[0]
    assert cmd[:len(expected)] == expected and cmd[len(expected)] == str(tmp_path / "prdm9")
    assert cmd[cmd.index("-f") + 1] == str(cli.ENVS["prdm9"])
    assert ("MAMBA_ROOT_PREFIX" in kw["env"]) == manager.endswith("micromamba")


def test_existing_env_is_kept_without_update(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ENV_DIR", tmp_path)
    (tmp_path / "prdm9").mkdir()
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("should not run"))
    cli.create_env("/x/mamba", "prdm9", update=False)


def test_snakemake_command(monkeypatch, tmp_path):
    calls = []
    smk = tmp_path / "prdm9" / "bin" / "snakemake"
    smk.parent.mkdir(parents=True)
    smk.touch()
    monkeypatch.setattr(cli, "ENV_DIR", tmp_path)
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda cmd, **kw: calls.append((cmd, kw)) or subprocess.CompletedProcess(cmd, 0))
    assert cli.main(["run", "fig2a", "--cores", "8", "--dry-run", "--config", "threads=2"]) == 0
    cmd, kw = calls[-1]
    assert cmd[:2] == [str(smk), "fig2a"]
    assert cmd[cmd.index("--cores") + 1] == "8" and "--dry-run" in cmd
    assert cmd[cmd.index("--config") + 1] == "threads=2"
    assert cmd[cmd.index("--directory") + 1] == str(REPO)
    assert kw["env"]["PATH"].startswith(str(tmp_path / "prdm9" / "bin"))
    assert cli.main(["fetch"]) == 0
    cmd, _ = calls[-1]
    assert cmd[1] == "data" and cmd[cmd.index("--cores") + 1] == cli.read_config_value("threads", "?")


def test_missing_env_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ENV_DIR", tmp_path)
    with pytest.raises(SystemExit, match="setup"):
        cli.main(["run"])


def test_env_specs_cover_required_imports():
    spec = yaml.safe_load((REPO / "envs" / "prdm9.yml").read_text())
    names = set()
    for dep in spec["dependencies"]:
        if isinstance(dep, dict):
            names |= {p.split("==")[0].lower() for p in dep["pip"]}
        else:
            names.add(dep.split("=")[0].lower())
    import_to_pkg = {"pyBigWig": "pybigwig", "yaml": "pyyaml", "sklearn": "scikit-learn",
                     "snakemake": "snakemake-minimal"}
    for mod in cli.REQUIRED_IMPORTS:
        assert import_to_pkg.get(mod, mod).lower() in names, mod
    for dep in spec["dependencies"]:
        if isinstance(dep, str) and dep.split("=")[0] not in ("pip",):
            assert "=" in dep, f"unpinned: {dep}"


def test_help_lists_all_commands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for c in ("doctor", "setup", "fetch", "run", "status", "test"):
        assert c in out
