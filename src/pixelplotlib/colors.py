"""Color parsing, palette snapping and bit-depth quantization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

RGB = tuple[int, int, int]
ColorLike = str | Sequence[float] | Sequence[int] | RGB

NAMED_COLORS: dict[str, RGB] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (228, 32, 32),
    "green": (32, 192, 48),
    "blue": (32, 80, 240),
    "yellow": (248, 216, 32),
    "cyan": (32, 216, 232),
    "magenta": (224, 48, 224),
    "orange": (248, 136, 24),
    "purple": (128, 48, 200),
    "pink": (248, 128, 176),
    "brown": (136, 80, 24),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "lightgray": (192, 192, 192),
    "lightgrey": (192, 192, 192),
    "darkgray": (72, 72, 72),
    "darkgrey": (72, 72, 72),
    "navy": (16, 24, 112),
    "teal": (0, 136, 136),
    "lime": (152, 248, 24),
    "olive": (128, 128, 0),
    "maroon": (128, 16, 16),
    "gold": (248, 184, 0),
    "sky": (60, 188, 252),
}


def parse_color(color: ColorLike) -> RGB:
    """Convert a color specification to an ``(r, g, b)`` tuple of ints.

    Accepts ``"#rgb"``/``"#rrggbb"`` hex strings, named colors and sequences of
    three numbers (floats in ``[0, 1]`` or ints in ``[0, 255]``).
    """
    if isinstance(color, str):
        s = color.strip().lower()
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            if len(h) != 6:
                raise ValueError(f"invalid hex color {color!r}")
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        key = s.replace(" ", "").replace("_", "")
        if key in NAMED_COLORS:
            return NAMED_COLORS[key]
        raise ValueError(f"unknown color {color!r}")
    vals = list(color)
    if len(vals) not in (3, 4):
        raise ValueError(f"invalid color {color!r}")
    vals = vals[:3]
    if all(isinstance(v, (float, np.floating)) for v in vals) and max(vals) <= 1.0:
        vals = [round(v * 255) for v in vals]
    return tuple(int(np.clip(v, 0, 255)) for v in vals)  # type: ignore[return-value]


def luminance(rgb: Sequence[float]) -> float:
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def mix(a: Sequence[int], b: Sequence[int], t: float) -> RGB:
    """Linearly interpolate between colors ``a`` and ``b``."""
    return tuple(int(round(float(x) + (float(y) - float(x)) * t))  # type: ignore
                 for x, y in zip(a[:3], b[:3]))


def shade(rgb: Sequence[int], amount: float) -> RGB:
    """Lighten (``amount > 0``) or darken (``amount < 0``) a color."""
    if amount >= 0:
        return mix(rgb, (255, 255, 255), amount)
    return mix(rgb, (0, 0, 0), -amount)


def _redmean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cheap perceptual color distance (squared), broadcasting over leading dims."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    rmean = (a[..., 0] + b[..., 0]) / 2.0
    d = a - b
    return (
        (2.0 + rmean / 256.0) * d[..., 0] ** 2
        + 4.0 * d[..., 1] ** 2
        + (2.0 + (255.0 - rmean) / 256.0) * d[..., 2] ** 2
    )


class ColorSystem:
    """Restricts colors to a hardware palette or to a per-channel bit depth.

    Parameters
    ----------
    palette:
        Allowed colors. Arbitrary colors are snapped to the nearest entry.
    bits:
        Bits per channel when no palette is given (5 bits = 15-bit "16-bit era" color).
    """

    def __init__(self, palette: Sequence[ColorLike] | None = None, bits: int | None = None):
        self.palette = None if palette is None else np.array(
            [parse_color(c) for c in palette], dtype=np.uint8
        )
        self.bits = bits

    def snap(self, color: ColorLike) -> RGB:
        rgb = np.array(parse_color(color), dtype=np.uint8)
        return tuple(int(v) for v in self.snap_array(rgb[None, :])[0])  # type: ignore

    def snap_array(self, arr: np.ndarray) -> np.ndarray:
        """Snap an ``(..., 3)`` uint8 array of colors."""
        arr = np.asarray(arr)
        if self.palette is not None:
            flat = arr.reshape(-1, 3)
            out = np.empty_like(flat, dtype=np.uint8)
            # chunk to keep memory bounded for large images
            for start in range(0, len(flat), 65536):
                chunk = flat[start : start + 65536]
                d = _redmean_dist(chunk[:, None, :], self.palette[None, :, :])
                out[start : start + 65536] = self.palette[np.argmin(d, axis=1)]
            return out.reshape(arr.shape)
        if self.bits is not None and self.bits < 8:
            return quantize_bits(arr, self.bits)
        return arr.astype(np.uint8)


def quantize_bits(arr: np.ndarray, bits: int) -> np.ndarray:
    """Reduce channels to ``bits`` bits, then expand back to the full 8-bit range."""
    arr = np.asarray(arr, dtype=np.uint8)
    levels = (1 << bits) - 1
    q = arr.astype(np.uint16) >> (8 - bits)
    return ((q * 255 + levels // 2) // levels).astype(np.uint8)
