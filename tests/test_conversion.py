"""The converted notebook stages must be current, compile, and contain exactly the documented changes."""
from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "tools"))
import convert_notebooks as conv  # noqa: E402

STAGE_FILES = sorted((REPO / "prdm9_pipeline" / "stages").glob("s*.py"))


def test_all_notebooks_have_a_stage():
    notebooks = {p.stem for p in (REPO / "notebooks").glob("*.ipynb")}
    assert notebooks == set(conv.STAGES)
    assert {p.stem for p in STAGE_FILES} == set(conv.STAGES.values())


def test_stage_files_are_up_to_date():
    result = subprocess.run([sys.executable, str(REPO / "tools" / "convert_notebooks.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("path", STAGE_FILES, ids=lambda p: p.stem)
def test_stage_compiles_and_has_no_magics(path):
    source = path.read_text()
    compile(source, str(path), "exec")
    assert "!pip" not in source
    assert "from IPython" not in source
    tree = ast.parse(source)
    futures = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    # __future__ imports must directly follow the module docstring
    for node in futures:
        assert tree.body.index(node) <= len(futures)


def test_code_cells_preserved_verbatim_apart_from_rules():
    """Every code line of the notebook survives except the lines the converter documents as rewritten."""
    for nb_name, module in conv.STAGES.items():
        nb = json.loads((REPO / "notebooks" / f"{nb_name}.ipynb").read_text())
        stage = (REPO / "prdm9_pipeline" / "stages" / f"{module}.py").read_text()
        stage_lines = set(stage.splitlines())
        rewritten = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            for line in conv.cell_text(cell).splitlines():
                if line not in stage_lines:
                    rewritten += 1
        # threads, iterations, pip, IPython import, future imports: a handful per notebook
        assert rewritten <= 6, f"{nb_name}: {rewritten} lines differ from the notebook"


def test_parameters_are_overridable():
    for module in conv.STAGES.values():
        src = (REPO / "prdm9_pipeline" / "stages" / f"{module}.py").read_text()
        assert "threads(" in src, module
    assert "mc_iters(200_000)" in (REPO / "prdm9_pipeline/stages/s3a_enrichment_g4.py").read_text()
    assert "mc_iters(200_000)" in (REPO / "prdm9_pipeline/stages/s3c_enrichment_all.py").read_text()
    assert "n_boot(5000)" in (REPO / "prdm9_pipeline/stages/s3b_enrichment_g4seq.py").read_text()
