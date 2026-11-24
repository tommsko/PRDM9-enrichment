"""Static checks for code that unit tests cannot execute end to end (the stages need the full data)."""
from __future__ import annotations

import pytest

from conftest import REPO

pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes import messages  # noqa: E402

FATAL = (messages.UndefinedName, messages.UndefinedLocal, messages.UndefinedExport, messages.ImportStarUsage,
         messages.ReturnOutsideFunction, messages.DuplicateArgument)
FILES = sorted([*(REPO / "prdm9_pipeline").rglob("*.py"), REPO / "prdm9.py"])


class Collect:
    def __init__(self):
        self.found = []

    def flake(self, msg):
        if isinstance(msg, FATAL):
            self.found.append(str(msg))

    def unexpectedError(self, filename, msg):
        self.found.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.found.append(f"{filename}:{lineno}: {msg}")


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_undefined_names(path):
    rep = Collect()
    pyflakes_api.checkPath(str(path), rep)
    assert not rep.found, "\n".join(rep.found)
