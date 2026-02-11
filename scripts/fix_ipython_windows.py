#!/usr/bin/env python3
"""Patch IPython profile config for safer Windows startup behavior.

This script updates (or creates) ``~/.ipython/profile_default/ipython_config.py`` to:
1) force UTF-8 mode (``PYTHONUTF8=1``) for child startup paths,
2) disable ``deduperreload`` auto-loading if present,
3) ensure ``IPython.extensions.autoreload`` is present in ``c.InteractiveShellApp.extensions``.

It is idempotent and can be run multiple times.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER_START = "# >>> MHDTurbPy IPython Windows fix >>>"
MARKER_END = "# <<< MHDTurbPy IPython Windows fix <<<"

BLOCK = f"""{MARKER_START}
import os
os.environ.setdefault('PYTHONUTF8', '1')

c = get_config()
_ext = list(getattr(c.InteractiveShellApp, 'extensions', []))
_ext = [e for e in _ext if e != 'deduperreload' and e != 'IPython.extensions.deduperreload']
if 'IPython.extensions.autoreload' not in _ext:
    _ext.append('IPython.extensions.autoreload')
c.InteractiveShellApp.extensions = _ext
{MARKER_END}
"""


def patch_config(path: Path) -> tuple[bool, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = ""

    start = text.find(MARKER_START)
    end = text.find(MARKER_END)
    if start != -1 and end != -1 and end > start:
        end_line = text.find("\n", end)
        if end_line == -1:
            end_line = len(text)
        else:
            end_line += 1
        text = text[:start] + text[end_line:]

    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + BLOCK

    path.write_text(text, encoding="utf-8")
    return True, str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch IPython config for Windows deduperreload/autoreload issues.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".ipython" / "profile_default" / "ipython_config.py",
        help="Target ipython_config.py path (default: ~/.ipython/profile_default/ipython_config.py)",
    )
    args = parser.parse_args()

    _, cfg = patch_config(args.config)
    print(f"Patched: {cfg}")
    print("Next steps:")
    print("  1) Restart IPython/Jupyter kernels.")
    print("  2) In a fresh session, run: %load_ext IPython.extensions.autoreload")
    print("  3) Then run: %autoreload 2")


if __name__ == "__main__":
    main()
