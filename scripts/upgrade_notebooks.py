"""Pre-commit hook: upgrade .ipynb files to nbformat 4.5.

GitHub's notebook renderer requires nbformat >= 4.5, which mandates a unique
``id`` field on every cell. Notebooks created by older Jupyter versions default
to 4.4 and lack cell IDs, causing a render error on GitHub.
"""

from __future__ import annotations

import json
import secrets
import sys

import nbformat


def upgrade(path: str) -> bool:
    """Return True if the file was modified."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("nbformat_minor", 0) >= 5:
        return False

    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if "id" not in cell or not cell["id"]:
            cell["id"] = secrets.token_hex(4)
    nb.nbformat_minor = 5
    nbformat.write(nb, path)
    return True


def main() -> None:
    """Upgrade each notebook path passed as a CLI argument."""
    modified = [p for p in sys.argv[1:] if upgrade(p)]
    for path in modified:
        print(f"upgraded to nbformat 4.5: {path}")
    if modified:
        sys.exit(1)


if __name__ == "__main__":
    main()
