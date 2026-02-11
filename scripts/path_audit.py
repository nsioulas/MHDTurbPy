#!/usr/bin/env python3
"""Audit repo path references used in Python and notebook code.

Default mode checks project source/notebook *code cells* for unresolved paths while
skipping vendored trees (e.g., ``pyspedas``), tests, and scripts.

Use flags to widen the scan surface (scripts, tests, notebook outputs, pyspedas).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_PATH_RE = re.compile(
    r"REPO_ROOT\s*/\s*['\"]([^'\"]+)['\"](?:\s*/\s*['\"]([^'\"]+)['\"])?(?:\s*/\s*['\"]([^'\"]+)['\"])?"
)
QUOTED_RE = re.compile(r"(?P<q>['\"])(?P<val>[^'\"\n]{2,220})(?P=q)")

REPO_HINT_PREFIXES = ("functions/", "examples/", "pyspedas/", "assets/", "requirements/")
IGNORE_LITERALS = {
    "./solar_orbiter_data",
    "./psp_data",
    "./data/solar_orbiter_data",
    "./data/psp_data",
    "/Users/",          # literal prefixes used by regex docs/tooling
    "/Applications/",   # literal prefixes used by regex docs/tooling
    "/usr/local/",      # literal prefixes used by regex docs/tooling
}


@dataclass
class Finding:
    file: str
    line: int
    value: str
    reason: str


def _iter_notebook_code_lines(obj: dict) -> Iterable[str]:
    for cell in obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for ln in cell.get("source", []):
            if ln.lstrip().startswith(("%", "!")):
                continue
            yield ln
        yield "\n"


def _iter_notebook_output_lines(obj: dict) -> Iterable[str]:
    for cell in obj.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text")
            if isinstance(text, list):
                for ln in text:
                    yield ln
            elif isinstance(text, str):
                yield text


def _code_text(path: Path, include_notebook_outputs: bool = False) -> str:
    if path.suffix == ".py":
        return path.read_text(encoding="utf-8", errors="ignore")

    obj = json.loads(path.read_text(encoding="utf-8"))
    parts = list(_iter_notebook_code_lines(obj))
    if include_notebook_outputs:
        parts.append("\n# --- notebook outputs ---\n")
        parts.extend(_iter_notebook_output_lines(obj))
    return "".join(parts)


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _candidate_literals(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for m in QUOTED_RE.finditer(text):
        value = m.group("val").strip()
        if value.startswith(("http://", "https://")):
            continue
        if value in IGNORE_LITERALS:
            continue
        if value.startswith(("/Users/", "/Applications/", "/usr/local/", "./", "../")) or value.startswith(REPO_HINT_PREFIXES):
            out.append((m.start(), value))
    return out


def _should_skip(path: Path, *, include_scripts: bool, include_pyspedas: bool, include_tests: bool) -> bool:
    if any(p in {".git", ".ipynb_checkpoints", "__pycache__"} for p in path.parts):
        return True
    if not include_pyspedas and "pyspedas" in path.parts:
        return True
    if not include_scripts and "scripts" in path.parts:
        return True
    if not include_tests and (path.name.endswith("_test.py") or "tests" in path.parts):
        return True
    return False


def audit(
    root: Path,
    *,
    include_scripts: bool = False,
    include_pyspedas: bool = False,
    include_tests: bool = False,
    include_notebook_outputs: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".ipynb"}:
            continue
        if _should_skip(path, include_scripts=include_scripts, include_pyspedas=include_pyspedas, include_tests=include_tests):
            continue

        text = _code_text(path, include_notebook_outputs=include_notebook_outputs)

        for m in REPO_PATH_RE.finditer(text):
            rel = "/".join(g for g in m.groups() if g)
            if not (root / rel).exists():
                findings.append(Finding(str(path.relative_to(root)), _line(text, m.start()), rel, "REPO_ROOT path does not exist"))

        for off, value in _candidate_literals(text):
            resolved = Path(value).expanduser()
            if resolved.is_absolute():
                ok = resolved.exists()
            else:
                ok = (root / resolved).exists() or (path.parent / resolved).exists()
            if not ok:
                findings.append(Finding(str(path.relative_to(root)), _line(text, off), value, "literal path does not exist"))

    uniq = {(f.file, f.line, f.value, f.reason): f for f in findings}
    return sorted(uniq.values(), key=lambda f: (f.file, f.line, f.value))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--include-scripts", action="store_true", help="Scan files under scripts/")
    p.add_argument("--include-pyspedas", action="store_true", help="Scan vendored pyspedas tree")
    p.add_argument("--include-tests", action="store_true", help="Scan *_test.py and tests/ paths")
    p.add_argument("--include-notebook-outputs", action="store_true", help="Also scan notebook outputs (stderr/stdout text)")
    p.add_argument("--json", action="store_true", help="Emit findings as JSON")
    args = p.parse_args()

    root = Path(args.root).resolve()
    findings = audit(
        root,
        include_scripts=args.include_scripts,
        include_pyspedas=args.include_pyspedas,
        include_tests=args.include_tests,
        include_notebook_outputs=args.include_notebook_outputs,
    )

    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2))
    elif findings:
        print(f"Found {len(findings)} unresolved path references:")
        for f in findings:
            print(f"- {f.file}:{f.line}: {f.value} ({f.reason})")
    else:
        print("No unresolved path references found.")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
