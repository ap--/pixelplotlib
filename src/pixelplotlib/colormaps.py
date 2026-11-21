"""Colormaps and palette-aware color ramps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

from .colors import ColorLike, ColorSystem, luminance, parse_color
from .dither import threshold_map


class Colormap:
    """Piecewise-linear colormap defined by anchor colors."""

    def __init__(
        self,
        name: str,
        colors: Sequence[ColorLike],
        positions: Sequence[float] | None = None,
        explicit: bool = False,
    ):
        self.name = name
        self.colors = np.array([parse_color(c) for c in colors], dtype=np.float64)
        if positions is None:
            positions = np.linspace(0.0, 1.0, len(self.colors))
        self.positions = np.asarray(positions, dtype=np.float64)
        # explicit colormaps are user-provided color lists that should be used verbatim
        self.explicit = explicit

    def __call__(self, t) -> np.ndarray:
        """Map values in ``[0, 1]`` to float RGB in ``[0, 255]``."""
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        return np.stack(
            [np.interp(t, self.positions, self.colors[:, k]) for k in range(3)], axis=-1
        )

    def reversed(self) -> Colormap:
        name = self.name[:-2] if self.name.endswith("_r") else self.name + "_r"
        return Colormap(name, self.colors[::-1], 1.0 - self.positions[::-1], self.explicit)

    def __repr__(self) -> str:
        return f"Colormap({self.name!r})"


_CMAP_DATA: dict[str, list[str]] = {
    "viridis": ["#440154", "#472D7B", "#3B528B", "#2C728E", "#21918C",
                "#27AD81", "#5EC962", "#AADC32", "#FDE725"],
    "magma": ["#000004", "#1C1044", "#4F127B", "#812581", "#B5367A",
              "#E55064", "#FB8761", "#FEC287", "#FCFDBF"],
    "inferno": ["#000004", "#1F0C48", "#550F6D", "#88226A", "#BA3655",
                "#E35933", "#F98C0A", "#F9C932", "#FCFFA4"],
    "plasma": ["#0D0887", "#4C02A1", "#7E03A8", "#A92395", "#CC4778",
               "#E56B5D", "#F89441", "#FDC328", "#F0F921"],
    "coolwarm": ["#3B4CC0", "#8DB0FE", "#DDDDDD", "#F49A7B", "#B40426"],
    "fire": ["#000000", "#C00000", "#FFA000", "#FFFFFF"],
    "ice": ["#000000", "#0030A0", "#20C0F0", "#FFFFFF"],
    "gray": ["#000000", "#FFFFFF"],
    "rainbow": ["#6E40AA", "#2060FF", "#00C8E0", "#20D040", "#F8E000", "#FF8000", "#E01010"],
    "terrain": ["#333399", "#0099FF", "#00CC66", "#FFFF99", "#805C54", "#FFFFFF"],
    "ocean": ["#002010", "#003060", "#1060A0", "#40A0D0", "#C0F0FF"],
    "gameboy": ["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"],
    "pico": ["#000000", "#1D2B53", "#7E2553", "#AB5236", "#FF004D",
             "#FFA300", "#FFEC27", "#FFF1E8"],
}
_CMAP_POSITIONS: dict[str, list[float]] = {
    "terrain": [0.0, 0.15, 0.25, 0.5, 0.75, 1.0],
}

COLORMAPS: dict[str, Colormap] = {
    name: Colormap(name, cols, _CMAP_POSITIONS.get(name)) for name, cols in _CMAP_DATA.items()
}
COLORMAPS["grey"] = COLORMAPS["gray"]

CmapLike = Union[str, Colormap, Sequence[ColorLike]]


def get_cmap(cmap: CmapLike) -> Colormap:
    """Look up a colormap by name (``"_r"`` suffix reverses it) or build one from colors."""
    if isinstance(cmap, Colormap):
        return cmap
    if isinstance(cmap, str):
        name = cmap.lower()
        if name in COLORMAPS:
            return COLORMAPS[name]
        if name.endswith("_r") and name[:-2] in COLORMAPS:
            return COLORMAPS[name[:-2]].reversed()
        raise ValueError(f"unknown colormap {cmap!r}; choose from {sorted(COLORMAPS)}")
    return Colormap("custom", list(cmap), explicit=True)


def register_cmap(name: str, colors: Sequence[ColorLike], positions=None) -> Colormap:
    """Register a new named colormap."""
    cm = Colormap(name, colors, positions)
    COLORMAPS[name.lower()] = cm
    return cm


class Ramp:
    """A short list of displayable colors at positions in ``[0, 1]``.

    Values in between two ramp entries are rendered with ordered dithering,
    which is how 8/16-bit hardware faked smooth gradients.
    """

    def __init__(self, positions: np.ndarray, colors: np.ndarray):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.colors = np.asarray(colors, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.colors)

    def map(self, t: np.ndarray, dither: bool = True, x0: int = 0, y0: int = 0) -> np.ndarray:
        """Map a 2-D array of values in ``[0, 1]`` to an RGB image."""
        t = np.clip(np.nan_to_num(np.asarray(t, dtype=np.float64), nan=0.0), 0.0, 1.0)
        if len(self.colors) == 1:
            return np.broadcast_to(self.colors[0], t.shape + (3,)).copy()
        pos = self.positions
        idx = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(pos) - 2)
        frac = (t - pos[idx]) / np.maximum(pos[idx + 1] - pos[idx], 1e-12)
        if t.ndim == 2:
            thresh = threshold_map(t.shape, x0, y0) if dither else 0.5
        else:
            thresh = 0.5
        idx = idx + (frac > thresh)
        return self.colors[idx]

    def color_at(self, t: float) -> tuple[int, int, int]:
        """Nearest (undithered) ramp color for a scalar value."""
        rgb = self.map(np.array([[t]]), dither=False)[0, 0]
        return tuple(int(v) for v in rgb)  # type: ignore[return-value]


def build_ramp(cmap: CmapLike, colors: ColorSystem, levels: int = 16) -> Ramp:
    """Build a displayable :class:`Ramp` for ``cmap`` under the given color system."""
    cm = get_cmap(cmap)
    if cm.explicit:
        snapped = colors.snap_array(np.round(cm.colors).astype(np.uint8))
        return Ramp(np.linspace(0, 1, len(snapped)), snapped)

    pal = colors.palette
    if pal is not None and len(pal) <= 4:
        # tiny palettes: use every shade ordered by brightness
        order = np.argsort([luminance(c) for c in pal])
        ramp_cols = pal[order]
        if luminance(cm(0.0)) > luminance(cm(1.0)):
            ramp_cols = ramp_cols[::-1]
        return Ramp(np.linspace(0, 1, len(ramp_cols)), ramp_cols)

    t = np.linspace(0.0, 1.0, levels)
    samples = colors.snap_array(np.round(cm(t)).astype(np.uint8))
    # collapse runs of identical colors, placing each at the center of its run
    positions, cols = [], []
    start = 0
    for i in range(1, levels + 1):
        if i == levels or not np.array_equal(samples[i], samples[start]):
            positions.append(t[start:i].mean())
            cols.append(samples[start])
            start = i
    positions[0], positions[-1] = 0.0, 1.0
    if len(cols) == 1:
        positions = [0.0]
    return Ramp(np.array(positions), np.array(cols))
