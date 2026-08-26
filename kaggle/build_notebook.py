"""Turn a percent-format script into a Kaggle notebook.

The source of record is the `.py`: it is readable in a diff, and a notebook is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"


def split_cells(source: str) -> list[tuple[str, list[str]]]:
    cells: list[tuple[str, list[str]]] = []
    kind, body = "code", []
    for line in source.splitlines():
        if line.startswith(MARKER):
            if body:
                cells.append((kind, body))
            kind = "markdown" if line.startswith(MARKDOWN_MARKER) else "code"
            body = []
            continue
        body.append(line)
    if body:
        cells.append((kind, body))
    return cells


def strip_comment_prefix(lines: list[str]) -> list[str]:
    return [
        line[2:] if line.startswith("# ") else line.removeprefix("#") for line in lines
    ]


def to_notebook(source: str) -> dict:
    cells = []
    for kind, body in split_cells(source):
        text = strip_comment_prefix(body) if kind == "markdown" else body
        while text and not text[0].strip():
            text.pop(0)
        while text and not text[-1].strip():
            text.pop()
        if not text:
            continue
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": [f"{line}\n" for line in text[:-1]] + [text[-1]],
        }
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    source_path = Path(sys.argv[1])
    target_path = Path(sys.argv[2])
    notebook = to_notebook(source_path.read_text())
    target_path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"{len(notebook['cells'])} cells -> {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
