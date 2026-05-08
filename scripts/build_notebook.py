"""Helper to build a Jupyter notebook from a list of (kind, source) cells.

Usage:

    from scripts.build_notebook import write_notebook
    write_notebook(
        "notebooks/01_eda_and_cleaning.ipynb",
        [("markdown", "# Title"), ("code", "print('hello')")],
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nbformat


def write_notebook(out_path: str | Path, cells: Iterable[tuple[str, str]]) -> None:
    nb = nbformat.v4.new_notebook()
    nb_cells = []
    for kind, src in cells:
        if kind == "markdown":
            nb_cells.append(nbformat.v4.new_markdown_cell(src))
        elif kind == "code":
            nb_cells.append(nbformat.v4.new_code_cell(src))
        else:
            raise ValueError(f"Unknown cell kind: {kind!r}")
    nb["cells"] = nb_cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
