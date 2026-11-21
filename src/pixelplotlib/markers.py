"""Pixel sprites used as scatter/line markers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_SPRITES: dict[str, list[str]] = {
    "p": ["#"],
    ".": [".#.", "###", ".#."],
    "o": [".###.", "#####", "#####", "#####", ".###."],
    "s": ["#####", "#####", "#####", "#####", "#####"],
    "^": ["...#...", "..###..", ".#####.", "#######"],
    "v": ["#######", ".#####.", "..###..", "...#..."],
    "D": ["...#...", "..###..", ".#####.", "#######", ".#####.", "..###..", "...#..."],
    "+": ["..#..", "..#..", "#####", "..#..", "..#.."],
    "x": ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "*": ["..#..", "#.#.#", ".###.", "#.#.#", "..#.."],
    "heart": [".##.##.", "#######", "#######", ".#####.", "..###..", "...#..."],
    "star": ["...#...", "..###..", "#######", ".#####.", "..###..", ".##.##.", "#.....#"],
    "skull": [".#####.", "#######", "#..#..#", "#######", "###.###", ".#####.", ".#.#.#."],
    "ufo": ["..###..", ".#####.", "#.#.#.#", "#######", ".#...#."],
    "drop": ["..#..", "..#..", ".###.", "#####", "#####", ".###."],
}
_ALIASES = {"circle": "o", "square": "s", "diamond": "D", "triangle": "^", "pixel": "p",
            "point": ".", "plus": "+", "cross": "x"}

MarkerLike = str | Sequence[str] | np.ndarray

MARKERS = tuple(_SPRITES) + tuple(_ALIASES)


def marker_mask(marker: MarkerLike, size: int = 1) -> np.ndarray:
    """Return the boolean sprite for ``marker`` scaled by integer factor ``size``.

    ``marker`` may be a name (see :data:`MARKERS`), a list of ``#``/``.`` row
    strings, or a boolean array.
    """
    if isinstance(marker, str):
        name = _ALIASES.get(marker, marker)
        if name not in _SPRITES:
            raise ValueError(f"unknown marker {marker!r}; choose from {sorted(MARKERS)}")
        rows = _SPRITES[name]
        mask = np.array([[c == "#" for c in r] for r in rows], dtype=bool)
    elif isinstance(marker, np.ndarray):
        mask = marker.astype(bool)
    else:
        mask = np.array([[c == "#" for c in r] for r in marker], dtype=bool)
    size = max(int(size), 1)
    if size > 1:
        mask = np.repeat(np.repeat(mask, size, axis=0), size, axis=1)
    return mask
