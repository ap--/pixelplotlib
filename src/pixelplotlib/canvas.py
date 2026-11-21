"""A tiny raster canvas with pixel-exact drawing primitives."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np

from .dither import BAYER4, threshold_map
from .fonts import BitmapFont

Color = Sequence[int]

DASH_PATTERNS: dict[str, tuple[int, ...] | None] = {
    "-": None,
    "solid": None,
    "--": (1, 1, 1, 1, 0, 0, 0),
    "dashed": (1, 1, 1, 1, 0, 0, 0),
    ":": (1, 0),
    "dotted": (1, 0),
    "-.": (1, 1, 1, 1, 0, 0, 1, 0, 0),
    "dashdot": (1, 1, 1, 1, 0, 0, 1, 0, 0),
    "none": (0,),
    "": (0,),
}


def dash_pattern(style) -> tuple[int, ...] | None:
    if style is None:
        return None
    if isinstance(style, str):
        try:
            return DASH_PATTERNS[style]
        except KeyError:
            raise ValueError(f"unknown linestyle {style!r}") from None
    return tuple(int(bool(v)) for v in style)


def _round(v: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(v, dtype=np.float64) + 0.5).astype(np.int64)


class Canvas:
    """An RGB pixel buffer with a clip rectangle.

    Coordinates are integer pixel positions; ``(0, 0)`` is the top-left pixel.
    Rectangles are given as ``(x0, y0, x1, y1)`` with exclusive ``x1``/``y1``.
    """

    def __init__(self, width: int, height: int, color: Color = (0, 0, 0)):
        self.width = int(width)
        self.height = int(height)
        self.pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.pixels[:] = color
        self._clip = (0, 0, self.width, self.height)

    # ------------------------------------------------------------------ clipping
    @contextmanager
    def clip(self, x0: int, y0: int, x1: int, y1: int) -> Iterator[None]:
        """Temporarily restrict drawing to a rectangle (intersected with the current clip)."""
        old = self._clip
        self._clip = (max(old[0], x0), max(old[1], y0), min(old[2], x1), min(old[3], y1))
        try:
            yield
        finally:
            self._clip = old

    def _visible(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cx0, cy0, cx1, cy1 = self._clip
        return (xs >= cx0) & (xs < cx1) & (ys >= cy0) & (ys < cy1)

    # ------------------------------------------------------------------ points
    def put(self, xs, ys, color: Color, alpha: float = 1.0) -> None:
        """Set individual pixels (vectorized)."""
        xs = np.asarray(xs, dtype=np.int64).ravel()
        ys = np.asarray(ys, dtype=np.int64).ravel()
        keep = self._visible(xs, ys)
        if alpha < 1.0:
            keep &= _thresh_at(xs, ys) < alpha
        self.pixels[ys[keep], xs[keep]] = color

    def put_rgb(self, xs, ys, rgb: np.ndarray) -> None:
        """Set pixels to per-pixel colors."""
        xs = np.asarray(xs, dtype=np.int64).ravel()
        ys = np.asarray(ys, dtype=np.int64).ravel()
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        keep = self._visible(xs, ys)
        self.pixels[ys[keep], xs[keep]] = rgb[keep]

    # ------------------------------------------------------------------ rectangles
    def _clip_rect(self, x0, y0, x1, y1) -> tuple[int, int, int, int]:
        cx0, cy0, cx1, cy1 = self._clip
        return max(int(x0), cx0), max(int(y0), cy0), min(int(x1), cx1), min(int(y1), cy1)

    def fill_rect(self, x0, y0, x1, y1, color: Color, alpha: float = 1.0) -> None:
        """Fill ``[x0, x1) x [y0, y1)``; ``alpha < 1`` uses an ordered dither pattern."""
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0, y0, x1, y1 = self._clip_rect(x0, y0, x1, y1)
        if x1 <= x0 or y1 <= y0:
            return
        if alpha >= 1.0:
            self.pixels[y0:y1, x0:x1] = color
        elif alpha > 0:
            mask = threshold_map((y1 - y0, x1 - x0), x0, y0) < alpha
            self.pixels[y0:y1, x0:x1][mask] = color

    def fill_mask(self, mask: np.ndarray, x: int, y: int, color: Color, alpha: float = 1.0):
        """Paint ``color`` wherever the boolean ``mask`` (placed at top-left ``x, y``) is set."""
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        x0, y0, x1, y1 = self._clip_rect(x, y, x + w, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sub = mask[y0 - y : y1 - y, x0 - x : x1 - x]
        if alpha < 1.0:
            sub = sub & (threshold_map(sub.shape, x0, y0) < alpha)
        self.pixels[y0:y1, x0:x1][sub] = color

    def blit(self, image: np.ndarray, x: int, y: int, mask: np.ndarray | None = None) -> None:
        """Copy an ``(h, w, 3)`` image with its top-left corner at ``(x, y)``."""
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self._clip_rect(x, y, x + w, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        src = image[y0 - y : y1 - y, x0 - x : x1 - x]
        if mask is None:
            self.pixels[y0:y1, x0:x1] = src
        else:
            m = mask[y0 - y : y1 - y, x0 - x : x1 - x]
            self.pixels[y0:y1, x0:x1][m] = src[m]

    def rect(self, x0, y0, x1, y1, color: Color) -> None:
        """1px outline of the rectangle ``[x0, x1) x [y0, y1)``."""
        self.fill_rect(x0, y0, x1, y0 + 1, color)
        self.fill_rect(x0, y1 - 1, x1, y1, color)
        self.fill_rect(x0, y0, x0 + 1, y1, color)
        self.fill_rect(x1 - 1, y0, x1, y1, color)

    def hline(self, x0, x1, y, color: Color, dash=None) -> None:
        self.polyline([x0, x1], [y, y], color, dash=dash)

    def vline(self, x, y0, y1, color: Color, dash=None) -> None:
        self.polyline([x, x], [y0, y1], color, dash=dash)

    # ------------------------------------------------------------------ lines
    def line(self, x0, y0, x1, y1, color: Color, width: int = 1, dash=None) -> None:
        self.polyline([x0, x1], [y0, y1], color, width=width, dash=dash)

    def polyline(
        self,
        xs,
        ys,
        color: Color,
        width: int = 1,
        dash=None,
        alpha: float = 1.0,
    ) -> None:
        """Draw connected line segments through float pixel coordinates.

        Non-finite coordinates break the line. Segments are clipped analytically
        before rasterization, so far off-screen points are cheap.
        """
        xs = np.asarray(xs, dtype=np.float64).ravel()
        ys = np.asarray(ys, dtype=np.float64).ravel()
        if xs.size == 0:
            return
        pattern = dash_pattern(dash)
        if xs.size == 1:
            xs = np.repeat(xs, 2)
            ys = np.repeat(ys, 2)
        ax0, ay0, ax1, ay1 = xs[:-1], ys[:-1], xs[1:], ys[1:]
        finite = np.isfinite(ax0) & np.isfinite(ay0) & np.isfinite(ax1) & np.isfinite(ay1)
        # pixel offset of each segment start along the whole polyline (keeps dashes continuous)
        steps = np.where(finite, np.maximum(np.abs(ax1 - ax0), np.abs(ay1 - ay0)), 0)
        seg_offset = np.concatenate([[0.0], np.cumsum(steps)[:-1]])

        pad = width + 1
        cx0, cy0, cx1, cy1 = self._clip
        t0, t1 = _liang_barsky(
            np.where(finite, ax0, 0), np.where(finite, ay0, 0),
            np.where(finite, ax1, 0), np.where(finite, ay1, 0),
            cx0 - pad, cy0 - pad, cx1 + pad, cy1 + pad,
        )
        keep = finite & (t0 <= t1)
        if not keep.any():
            return
        dx, dy = ax1 - ax0, ay1 - ay0
        sx0 = _round(ax0[keep] + t0[keep] * dx[keep])
        sy0 = _round(ay0[keep] + t0[keep] * dy[keep])
        sx1 = _round(ax0[keep] + t1[keep] * dx[keep])
        sy1 = _round(ay0[keep] + t1[keep] * dy[keep])
        offs = _round(seg_offset[keep] + t0[keep] * steps[keep])

        px, py, idx = _rasterize_segments(sx0, sy0, sx1, sy1, offs)
        if pattern is not None:
            on = np.asarray(pattern, dtype=bool)[idx % len(pattern)]
            px, py = px[on], py[on]
        self._stamp(px, py, color, width, alpha)

    def _stamp(self, px, py, color, width: int, alpha: float = 1.0) -> None:
        if width <= 1:
            self.put(px, py, color, alpha)
            return
        lo = -((width - 1) // 2)
        offs = np.arange(lo, lo + width)
        ox, oy = np.meshgrid(offs, offs)
        self.put(
            (px[:, None] + ox.ravel()[None, :]).ravel(),
            (py[:, None] + oy.ravel()[None, :]).ravel(),
            color,
            alpha,
        )

    # ------------------------------------------------------------------ polygons
    def polygon_mask(self, xs, ys) -> tuple[np.ndarray, int, int]:
        """Even-odd scanline coverage of a polygon; returns ``(mask, x0, y0)``."""
        xs = np.asarray(xs, dtype=np.float64).ravel()
        ys = np.asarray(ys, dtype=np.float64).ravel()
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        cx0, cy0, cx1, cy1 = self._clip
        if xs.size < 3:
            return np.zeros((0, 0), bool), cx0, cy0
        ymin = max(int(np.ceil(ys.min())), cy0)
        ymax = min(int(np.floor(ys.max())), cy1 - 1)
        if ymax < ymin:
            return np.zeros((0, 0), bool), cx0, cy0
        xa, ya = xs, ys
        xb, yb = np.roll(xs, -1), np.roll(ys, -1)
        rows = np.arange(ymin, ymax + 1, dtype=np.float64)[:, None]
        crosses = ((ya <= rows) & (yb > rows)) | ((yb <= rows) & (ya > rows))
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = xa + (rows - ya) * (xb - xa) / (yb - ya)
        width = cx1 - cx0
        cols = np.clip(np.ceil(xi - 1e-9).astype(np.int64) - cx0, 0, width)
        acc = np.zeros((rows.shape[0], width + 1), dtype=np.int32)
        r_idx = np.broadcast_to(np.arange(rows.shape[0])[:, None], cols.shape)
        np.add.at(acc, (r_idx[crosses], cols[crosses]), 1)
        mask = (np.cumsum(acc, axis=1)[:, :width] % 2).astype(bool)
        return mask, cx0, ymin

    def fill_polygon(self, xs, ys, color: Color, alpha: float = 1.0) -> None:
        mask, x0, y0 = self.polygon_mask(xs, ys)
        if mask.size:
            self.fill_mask(mask, x0, y0, color, alpha)

    # ------------------------------------------------------------------ circles
    @staticmethod
    def disc_mask(r: float) -> np.ndarray:
        """Boolean mask of a filled pixel disc with radius ``r`` (odd-sized, centered)."""
        n = int(np.ceil(r))
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        return xx * xx + yy * yy <= r * r + r * 0.8

    def circle(self, cx, cy, r: float, color: Color, fill: bool = False, alpha: float = 1.0):
        m = self.disc_mask(r)
        if not fill:
            m = m & ~_erode(m)
        n = m.shape[0] // 2
        self.fill_mask(m, int(round(cx)) - n, int(round(cy)) - n, color, alpha)

    # ------------------------------------------------------------------ text
    def text(
        self,
        x: int,
        y: int,
        text: str,
        font: BitmapFont,
        color: Color,
        ha: str = "left",
        va: str = "top",
        rotation: int = 0,
        shadow: Color | None = None,
        align: str | None = None,
    ) -> tuple[int, int, int, int]:
        """Draw text anchored at ``(x, y)``; returns the bounding box ``(x0, y0, x1, y1)``.

        ``va="center"`` centers on the capital letters (ignores descenders).
        ``rotation`` may be 0, 90 (reads bottom-to-top) or 270.
        """
        if text is None or str(text) == "":
            return (x, y, x, y)
        align = align or {"right": "right", "center": "center"}.get(ha, "left")
        mask = font.render(str(text), align)
        if rotation % 360:
            mask = np.rot90(mask, k=(rotation // 90) % 4)
        h, w = mask.shape
        if rotation % 360 == 0:
            cap_h = h - font.descent
            vy = {"top": 0, "center": cap_h // 2, "baseline": cap_h - 1,
                  "bottom": h - 1}[va]
        else:
            vy = {"top": 0, "center": h // 2, "bottom": h - 1, "baseline": h - 1}[va]
        hx = {"left": 0, "center": w // 2, "right": w - 1}[ha]
        x0, y0 = int(x) - hx, int(y) - vy
        if shadow is not None:
            self.fill_mask(mask, x0 + 1, y0 + 1, shadow)
        self.fill_mask(mask, x0, y0, color)
        return (x0, y0, x0 + w, y0 + h)

    # ------------------------------------------------------------------ export
    def to_array(self, scale: int = 1) -> np.ndarray:
        """Return the pixels, upscaled with nearest-neighbour by an integer factor."""
        if scale == 1:
            return self.pixels.copy()
        return np.repeat(np.repeat(self.pixels, scale, axis=0), scale, axis=1)


def _thresh_at(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return BAYER4[ys % 4, xs % 4]


def _erode(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1)
    return p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def dilate(m: np.ndarray, diagonal: bool = False) -> np.ndarray:
    """Grow a boolean mask by one pixel (output is 2px larger in each dimension)."""
    p = np.pad(m, 2)
    out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    if diagonal:
        out |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return out


def _liang_barsky(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    dx = x1 - x0
    dy = y1 - y0
    t0 = np.zeros_like(x0)
    t1 = np.ones_like(x0)
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = q / p
        neg = p < 0
        pos = p > 0
        t0 = np.where(neg, np.maximum(t0, r), t0)
        t1 = np.where(pos, np.minimum(t1, r), t1)
        # parallel and outside: reject
        t1 = np.where((p == 0) & (q < 0), -1.0, t1)
    return t0, t1


def _rasterize_segments(x0, y0, x1, y1, offsets):
    """DDA rasterization of many integer segments at once.

    Returns pixel x, y and the running pixel index (for dash patterns).
    """
    dx = x1 - x0
    dy = y1 - y0
    n = np.maximum(np.abs(dx), np.abs(dy))
    count = n + 1
    seg = np.repeat(np.arange(len(n)), count)
    starts = np.cumsum(count) - count
    k = np.arange(count.sum()) - np.repeat(starts, count)
    denom = np.maximum(n[seg], 1)
    px = x0[seg] + _round_div(dx[seg] * k, denom)
    py = y0[seg] + _round_div(dy[seg] * k, denom)
    return px, py, offsets[seg] + k


def _round_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer division rounding half away from zero (symmetric for both directions)."""
    q = (2 * np.abs(a) + b) // (2 * b)
    return np.sign(a) * q
