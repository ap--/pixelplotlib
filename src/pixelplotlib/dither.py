"""Ordered (Bayer) dithering helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .colors import ColorLike, ColorSystem


def bayer_matrix(n: int = 4) -> np.ndarray:
    """Return an ``n x n`` Bayer threshold matrix with values in ``(0, 1)``."""
    if n < 2 or n & (n - 1):
        raise ValueError("n must be a power of two >= 2")
    m = np.array([[0, 2], [3, 1]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


BAYER4 = bayer_matrix(4)


def threshold_map(shape: tuple[int, int], x0: int = 0, y0: int = 0, n: int = 4) -> np.ndarray:
    """Tile the Bayer matrix over ``shape``, anchored at absolute canvas position ``(x0, y0)``.

    Anchoring at absolute coordinates keeps patterns seamless between adjacent regions.
    """
    m = BAYER4 if n == 4 else bayer_matrix(n)
    h, w = shape
    ys = (np.arange(h) + y0) % m.shape[0]
    xs = (np.arange(w) + x0) % m.shape[1]
    return m[ys[:, None], xs[None, :]]


def dither_mask(shape: tuple[int, int], level: float, x0: int = 0, y0: int = 0) -> np.ndarray:
    """Boolean mask whose density of ``True`` pixels approximates ``level`` in ``[0, 1]``."""
    return threshold_map(shape, x0, y0) < level


def pixelate(
    image: np.ndarray,
    palette: Sequence[ColorLike] | None = None,
    size: tuple[int, int] | None = None,
    dither: bool = True,
    bits: int | None = None,
    strength: float | None = None,
) -> np.ndarray:
    """Turn an RGB image into pixel art.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` array, uint8 or float in ``[0, 1]``.
    palette:
        Target palette (list of colors). If ``None``, ``bits`` per channel is used.
    size:
        Optional output ``(width, height)``; nearest-neighbour downsampling is used.
    dither:
        Apply ordered dithering before snapping colors.
    bits:
        Bits per channel when no palette is given (default 5).
    strength:
        Dither amplitude in 8-bit units; defaults to a value derived from the palette.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3]
    if img.dtype.kind == "f":
        img = np.clip(img * 255.0, 0, 255)
    img = img.astype(np.float64)
    if size is not None:
        w, h = size
        ys = (np.arange(h) * img.shape[0] / h).astype(int)
        xs = (np.arange(w) * img.shape[1] / w).astype(int)
        img = img[ys[:, None], xs[None, :]]
    system = ColorSystem(palette=palette, bits=None if palette is not None else (bits or 5))
    if dither:
        if strength is None:
            if system.palette is not None:
                strength = 256.0 / max(np.cbrt(len(system.palette)), 1.0)
            else:
                strength = 256.0 / (1 << system.bits)
        t = threshold_map(img.shape[:2]) - 0.5
        img = img + t[..., None] * strength
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return system.snap_array(img)
