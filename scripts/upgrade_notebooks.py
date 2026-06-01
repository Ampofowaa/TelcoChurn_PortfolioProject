"""Pre-commit hook: upgrade .ipynb files to nbformat 4.5.

GitHub's notebook renderer requires nbformat >= 4.5, which mandates a unique
``id`` field on every cell. Notebooks created by older Jupyter versions default
to 4.4 and lack cell IDs, causing a render error on GitHub.
"""

from __future__ import annotations

import secrets
import sys

import nbformat


def upgrade(path: str) -> bool:
    """Return True if the file was modified."""
    nb = nbformat.read(path, as_version=4)

    needs_minor_bump = nb.nbformat_minor < 5
    needs_id_fix = any(len(cell.get("id", "")) < 8 for cell in nb.cells)
    if not needs_minor_bump and not needs_id_fix:
        return False

    for cell in nb.cells:
        if len(cell.get("id", "")) < 8:
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
