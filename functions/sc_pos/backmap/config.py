from __future__ import annotations

"""sc_pos.backmap.config

Small, explicit helpers for dict-driven configuration.

The intended usage is to define several dictionaries (grouped by category) and
combine them into a single config dict that is passed to the runner.

We provide two merge styles:

1) ``merge_dicts`` (shallow): last-write-wins for top-level keys.
2) ``deep_merge`` (recursive): merges nested dicts instead of overwriting them.

These utilities are deliberately minimal and dependency-free.
"""

from typing import Any, Dict, Mapping, MutableMapping, Optional


def merge_dicts(*dicts: Mapping[str, Any]) -> Dict[str, Any]:
    """Shallow merge (last-write-wins).

    Parameters
    ----------
    *dicts:
        Any number of mapping objects.

    Returns
    -------
    dict
        A new dict containing the union of keys.
    """
    out: Dict[str, Any] = {}
    for d in dicts:
        if d is None:
            continue
        out.update(dict(d))
    return out


def _deep_merge_into(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            _deep_merge_into(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v


def deep_merge(*dicts: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (nested dicts are merged, not overwritten)."""
    out: Dict[str, Any] = {}
    for d in dicts:
        if d is None:
            continue
        _deep_merge_into(out, dict(d))
    return out
