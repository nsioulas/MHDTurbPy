#!/usr/bin/env python3
"""Audit repo path references used in Python and notebook code."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_PATH_RE = re.compile(
    r"REPO_ROOT\s*/\s*['\"]([^'\"]+)['\"](?:\s*/\s*['\"]([^'\"]+)['\"])?(?:\s*/\s*['\"]([^'\"]+)['\"])?"
)
QUOTED_RE = re.compile(r"(?P<q>['\"])(?P<val>[^'\"\n]{2,220})(?P=q)")

REPO_HINT_PREFIXES = ("functions/", "examples/", "pyspedas/", "assets/", "requirements/")
IGNORE_LITERALS = {"./solar_orbiter_data", "./psp_data"}


@dataclass
class Finding:
    file: str
    line: int
    value: str
    reason: str


def _code_text(path: Path) -> str:
    if path.suffix == ".py":
        return path.read_text(encoding="utf-8", errors="ignore")

    obj = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for ln in cell.get("source", []):
            if ln.lstrip().startswith(("%", "!")):
                continue
            parts.append(ln)
        parts.append("\n")
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


def audit(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".ipynb"}:
            continue
        if any(p in {".git", ".ipynb_checkpoints", "__pycache__"} for p in path.parts):
            continue
        if "pyspedas" in path.parts or "scripts" in path.parts:
            continue
        if path.name.endswith("_test.py"):
            continue

        text = _code_text(path)

        for m in REPO_PATH_RE.finditer(text):
            rel = "/".join(g for g in m.groups() if g)
            if not (root / rel).exists():
                findings.append(Finding(str(path.relative_to(root)), _line(text, m.start()), rel, "REPO_ROOT path does not exist"))

        for off, value in _candidate_literals(text):
            resolved = Path(value).expanduser()
            ok = False
            if resolved.is_absolute():
                ok = resolved.exists()
            else:
                ok = (root / resolved).exists() or (path.parent / resolved).exists()
            if not ok:
                findings.append(Finding(str(path.relative_to(root)), _line(text, off), value, "literal path does not exist"))

    # deduplicate
    uniq = {(f.file, f.line, f.value, f.reason): f for f in findings}
    return sorted(uniq.values(), key=lambda f: (f.file, f.line, f.value))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    args = p.parse_args()

    root = Path(args.root).resolve()
    findings = audit(root)
    if findings:
        print(f"Found {len(findings)} unresolved path references:")
        for f in findings:
            print(f"- {f.file}:{f.line}: {f.value} ({f.reason})")
        return 1

    print("No unresolved path references found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
