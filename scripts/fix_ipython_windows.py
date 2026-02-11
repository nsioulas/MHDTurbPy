#!/usr/bin/env python3
"""Patch IPython profile config for safer Windows startup behavior.

This script updates (or creates) ``~/.ipython/profile_default/ipython_config.py`` to:
1) force UTF-8 mode (``PYTHONUTF8=1``),
2) disable ``deduperreload`` auto-loading if present,
3) ensure ``IPython.extensions.autoreload`` is present.

It also patches startup files under ``.../profile_default/startup`` to replace
``%load_ext autoreload`` with the fully-qualified extension name and to disable
``deduperreload`` extension loads.

The script is idempotent and can be run multiple times.
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
_ext = [e for e in _ext if e not in ('deduperreload', 'IPython.extensions.deduperreload')]
if 'IPython.extensions.autoreload' not in _ext:
    _ext.append('IPython.extensions.autoreload')
c.InteractiveShellApp.extensions = _ext
{MARKER_END}
"""


def patch_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    start = text.find(MARKER_START)
    end = text.find(MARKER_END)
    if start != -1 and end != -1 and end > start:
        end_line = text.find("\n", end)
        end_line = len(text) if end_line == -1 else end_line + 1
        text = text[:start] + text[end_line:]

    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + BLOCK

    path.write_text(text, encoding="utf-8")
    return path


def patch_startup_files(profile_dir: Path) -> int:
    startup_dir = profile_dir / "startup"
    if not startup_dir.exists():
        return 0

    changed = 0
    for p in startup_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in {".py", ".ipy"}:
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")
        new = text
        new = new.replace("%load_ext autoreload", "%load_ext IPython.extensions.autoreload")
        new = new.replace("%load_ext deduperreload", "# disabled by MHDTurbPy fix: %load_ext deduperreload")
        new = new.replace(
            "%load_ext IPython.extensions.deduperreload",
            "# disabled by MHDTurbPy fix: %load_ext IPython.extensions.deduperreload",
        )

        if new != text:
            p.write_text(new, encoding="utf-8")
            changed += 1

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch IPython config for Windows deduperreload/autoreload issues.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".ipython" / "profile_default" / "ipython_config.py",
        help="Target ipython_config.py path (default: ~/.ipython/profile_default/ipython_config.py)",
    )
    args = parser.parse_args()

    cfg_path = patch_config(args.config)
    startup_changes = patch_startup_files(cfg_path.parent)

    print(f"Patched config: {cfg_path}")
    print(f"Patched startup files: {startup_changes}")
    print("Next steps:")
    print("  1) Restart IPython/Jupyter kernels.")
    print("  2) In a fresh session, run: %load_ext IPython.extensions.autoreload")
    print("  3) Then run: %autoreload 2")


if __name__ == "__main__":
    main()
