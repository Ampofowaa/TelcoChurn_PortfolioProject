"""Pre-commit hook: normalize fragmented stream outputs in executed notebooks.

On Windows, ipykernel's zmq fallback (Proactor event loop lacks add_reader
support) tends to flush each print()'s content and its trailing newline as
separate iopub messages. nbconvert --execute then saves each piece as its
own list element (or even its own adjacent ``stream`` output dict), instead
of one line-per-element the way a normal kernel run would. All shapes join
to the same final string, but some renderers (observed in VS Code) render
each ``text`` list element as its own line, so orphan newline-only elements
show up as blank-line gaps between prints. This hook flattens every stream
output's text and re-splits it into the canonical one-line-per-element form
(each element ending in its own ``\\n``, matching what nbformat/Jupyter
write for a normal, unfragmented run), merging adjacent same-stream outputs
in the process.
"""

from __future__ import annotations

import copy
import sys

import nbformat


def _flatten(text: str | list) -> str:  # type: ignore[type-arg]
    """nbformat's MultilineString type allows a stream's ``text`` to be a str or list[str]."""
    return text if isinstance(text, str) else "".join(text)


def _normalize(outputs: list) -> list:  # type: ignore[type-arg]
    """Merge adjacent same-stream outputs, in place, flattening each stream's text to one string.

    nbformat.read() already loads ``text`` as a joined string regardless of its
    on-disk str/list form, and nbformat.write() re-splits it into the canonical
    one-line-per-element list at serialization time — reproducing that split here
    would just create a spurious str-vs-list diff against the freshly-read cell.
    """
    merged: list = []  # type: ignore[type-arg]
    for output in outputs:
        prev = merged[-1] if merged else None
        if (
            output.get("output_type") == "stream"
            and prev is not None
            and prev.get("output_type") == "stream"
            and prev.get("name") == output.get("name")
        ):
            prev["text"] = _flatten(prev["text"]) + _flatten(output.get("text", ""))
        else:
            merged.append(output)
    for output in merged:
        if output.get("output_type") == "stream":
            output["text"] = _flatten(output["text"])
    return merged


def fix(path: str) -> bool:
    """Return True if the file was modified."""
    nb = nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]

    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code" or not cell.get("outputs"):
            continue
        before = copy.deepcopy(cell["outputs"])
        cell["outputs"] = _normalize(cell["outputs"])
        if cell["outputs"] != before:
            changed = True

    if changed:
        nbformat.write(nb, path)  # type: ignore[no-untyped-call]
    return changed


def main() -> None:
    """Fix each notebook path passed as a CLI argument."""
    modified = [p for p in sys.argv[1:] if fix(p)]
    for path in modified:
        print(f"normalized fragmented stream outputs: {path}")
    if modified:
        sys.exit(1)


if __name__ == "__main__":
    main()
