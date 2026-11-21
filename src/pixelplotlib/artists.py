"""Drawable plot elements."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .canvas import Canvas, dash_pattern, dilate
from .colormaps import Ramp
from .dither import pixelate
from .markers import marker_mask

EPS = 1e-6


def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def _finite_range(v: np.ndarray, log: bool):
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if v.size == 0:
        return None
    return float(v.min()), float(v.max())


class Scale:
    """Maps data values to a linear "transformed" space (identity or log10)."""

    def __init__(self, kind: str = "linear"):
        if kind not in ("linear", "log"):
            raise ValueError("scale must be 'linear' or 'log'")
        self.kind = kind

    @property
    def log(self) -> bool:
        return self.kind == "log"

    def fwd(self, v):
        v = np.asarray(v, dtype=np.float64)
        if self.log:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
        return v

    def inv(self, u):
        u = np.asarray(u, dtype=np.float64)
        return 10.0**u if self.log else u


class Norm:
    """Value normalization for colormapped artists."""

    def __init__(self, vmin: float, vmax: float, kind: str = "linear"):
        self.vmin, self.vmax, self.kind = float(vmin), float(vmax), kind
        if kind == "log" and (self.vmin <= 0 or self.vmax <= 0):
            raise ValueError("log norm needs positive vmin/vmax")

    def __call__(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if self.kind == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                lv = np.log10(np.where(v > 0, v, np.nan))
            lo, hi = math.log10(self.vmin), math.log10(self.vmax)
            return (lv - lo) / ((hi - lo) or 1.0)
        return (v - self.vmin) / ((self.vmax - self.vmin) or 1.0)


def make_norm(values, vmin=None, vmax=None, norm: str | None = None) -> Norm:
    kind = norm or "linear"
    v = _f(values)
    v = v[np.isfinite(v)]
    if kind == "log":
        v = v[v > 0]
    lo = vmin if vmin is not None else (float(v.min()) if v.size else 0.0)
    hi = vmax if vmax is not None else (float(v.max()) if v.size else 1.0)
    if lo == hi:
        lo, hi = (lo / 10, hi * 10) if kind == "log" else (lo - 0.5, hi + 0.5)
    return Norm(lo, hi, kind)


class RenderContext:
    """Everything an artist needs to draw itself inside an axes."""

    def __init__(self, canvas: Canvas, style, rect, xlim, ylim, xscale: Scale, yscale: Scale):
        self.canvas = canvas
        self.style = style
        self.rect = rect
        self.xlim, self.ylim = xlim, ylim
        self.xscale, self.yscale = xscale, yscale
        self._fx = (float(xscale.fwd(xlim[0])), float(xscale.fwd(xlim[1])))
        self._fy = (float(yscale.fwd(ylim[0])), float(yscale.fwd(ylim[1])))

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    def ux(self, x):
        lo, hi = self._fx
        return (self.xscale.fwd(x) - lo) / ((hi - lo) or 1.0)

    def uy(self, y):
        lo, hi = self._fy
        return (self.yscale.fwd(y) - lo) / ((hi - lo) or 1.0)

    def tx(self, x):
        """Data x -> pixel-center x coordinate (float)."""
        return self.rect[0] - 0.5 + self.ux(x) * (self.width - EPS)

    def ty(self, y):
        return self.rect[1] - 0.5 + (1.0 - self.uy(y)) * (self.height - EPS)

    def inv_x(self, px):
        u = (np.asarray(px, dtype=np.float64) - (self.rect[0] - 0.5)) / (self.width - EPS)
        lo, hi = self._fx
        return self.xscale.inv(lo + u * (hi - lo))

    def inv_y(self, py):
        u = 1.0 - (np.asarray(py, dtype=np.float64) - (self.rect[1] - 0.5)) / (self.height - EPS)
        lo, hi = self._fy
        return self.yscale.inv(lo + u * (hi - lo))

    # pixel spans for areas: pixels whose centers lie within [a, b)
    @staticmethod
    def span(a: float, b: float, minimum: int = 0) -> tuple[int, int]:
        lo, hi = (a, b) if a <= b else (b, a)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 0, 0
        p0, p1 = int(math.ceil(lo)), int(math.ceil(hi))
        if p1 - p0 < minimum:
            if b < a:  # value grows upwards on screen: extend towards smaller pixel rows
                p0 = p1 - minimum
            else:
                p1 = p0 + minimum
        return p0, p1


class Artist:
    zorder = 2
    label: str | None = None
    sticky_x: Sequence[float] = ()
    sticky_y: Sequence[float] = ()

    def bounds(self, xlog: bool, ylog: bool):
        """Data bounding box ``(xmin, xmax, ymin, ymax)``; entries may be ``None``."""
        return None

    def points(self):
        """Representative data points (for legend placement)."""
        return None

    def legend_entries(self) -> list[dict]:
        return []

    def draw(self, ctx: RenderContext) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


def _xy_bounds(x, y, xlog, ylog):
    x, y = _f(x), _f(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if xlog:
        ok &= x > 0
    if ylog:
        ok &= y > 0
    if not ok.any():
        return None
    return float(x[ok].min()), float(x[ok].max()), float(y[ok].min()), float(y[ok].max())


# ---------------------------------------------------------------------- markers
def draw_markers(ctx: RenderContext, px, py, marker, size: int, colors, outline=None):
    """Stamp sprites at pixel-center coordinates. ``colors`` is one RGB or one per point."""
    px, py = _f(px), _f(py)
    ok = np.isfinite(px) & np.isfinite(py)
    if not ok.any():
        return
    cx = np.floor(px[ok] + 0.5).astype(np.int64)
    cy = np.floor(py[ok] + 0.5).astype(np.int64)
    mask = marker_mask(marker, size)
    h, w = mask.shape
    per_point = isinstance(colors, np.ndarray) and colors.ndim == 2
    if outline is not None:
        om = dilate(mask)
        oy, ox = np.nonzero(om)
        ctx.canvas.put(
            (cx[:, None] + ox[None, :] - 2 - w // 2).ravel(),
            (cy[:, None] + oy[None, :] - 2 - h // 2).ravel(),
            outline,
        )
    my, mx = np.nonzero(mask)
    xs = (cx[:, None] + mx[None, :] - w // 2).ravel()
    ys = (cy[:, None] + my[None, :] - h // 2).ravel()
    if per_point:
        cols = np.repeat(colors[ok], len(mx), axis=0)
        ctx.canvas.put_rgb(xs, ys, cols)
    else:
        ctx.canvas.put(xs, ys, colors)


# ---------------------------------------------------------------------- lines
class Line(Artist):
    def __init__(self, x, y, color, width=1, linestyle="-", marker=None, markersize=1,
                 label=None, shadow=True, zorder=2):
        self.x, self.y = _f(x), _f(y)
        self.color = color
        self.width = int(width)
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize
        self.label = label
        self.shadow = shadow
        self.zorder = zorder

    def bounds(self, xlog, ylog):
        return _xy_bounds(self.x, self.y, xlog, ylog)

    def points(self):
        return self.x, self.y

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="line", label=self.label, color=self.color, linestyle=self.linestyle,
                     width=self.width, marker=self.marker)]

    def draw(self, ctx):
        px, py = ctx.tx(self.x), ctx.ty(self.y)
        st = ctx.style
        if self.linestyle not in ("none", "", None):
            if self.shadow and st.line_halo is not None:
                # background-colored outline keeps lines readable on dithered fills
                ctx.canvas.polyline(px, py, st.color(st.line_halo), self.width + 2,
                                    self.linestyle)
            if self.shadow and st.line_shadow is not None:
                ctx.canvas.polyline(px + 1, py + 1, st.color(st.line_shadow), self.width,
                                    self.linestyle)
            ctx.canvas.polyline(px, py, self.color, self.width, self.linestyle)
        if self.marker:
            outline = st.color(st.marker_outline) if st.marker_outline is not None else None
            draw_markers(ctx, px, py, self.marker, self.markersize, self.color, outline)


class RefLine(Artist):
    """Infinite horizontal or vertical line (``axhline``/``axvline``)."""

    def __init__(self, value, orientation, color, linestyle="-", width=1, label=None):
        self.value = float(value)
        self.orientation = orientation
        self.color = color
        self.linestyle = linestyle
        self.width = width
        self.label = label
        self.zorder = 1.5

    def bounds(self, xlog, ylog):
        if self.orientation == "h":
            return None, None, self.value, self.value
        return self.value, self.value, None, None

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="line", label=self.label, color=self.color, linestyle=self.linestyle,
                     width=self.width, marker=None)]

    def draw(self, ctx):
        x0, y0, x1, y1 = ctx.rect
        if self.orientation == "h":
            y = float(ctx.ty(self.value))
            ctx.canvas.polyline([x0, x1 - 1], [y, y], self.color, self.width, self.linestyle)
        else:
            x = float(ctx.tx(self.value))
            ctx.canvas.polyline([x, x], [y0, y1 - 1], self.color, self.width, self.linestyle)


class Span(Artist):
    """Shaded band across the axes (``axhspan``/``axvspan``)."""

    def __init__(self, v0, v1, orientation, color, alpha=0.5, label=None):
        self.v0, self.v1 = float(v0), float(v1)
        self.orientation = orientation
        self.color = color
        self.alpha = alpha
        self.label = label
        self.zorder = 1

    def bounds(self, xlog, ylog):
        lo, hi = min(self.v0, self.v1), max(self.v0, self.v1)
        if self.orientation == "h":
            return None, None, lo, hi
        return lo, hi, None, None

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="patch", label=self.label, color=self.color, alpha=self.alpha)]

    def draw(self, ctx):
        x0, y0, x1, y1 = ctx.rect
        if self.orientation == "h":
            a, b = ctx.span(float(ctx.ty(self.v0)), float(ctx.ty(self.v1)))
            ctx.canvas.fill_rect(x0, a, x1, b, self.color, self.alpha)
        else:
            a, b = ctx.span(float(ctx.tx(self.v0)), float(ctx.tx(self.v1)))
            ctx.canvas.fill_rect(a, y0, b, y1, self.color, self.alpha)


# ---------------------------------------------------------------------- scatter
class Scatter(Artist):
    def __init__(self, x, y, color, marker="o", size=1, values=None, ramp: Ramp | None = None,
                 norm: Norm | None = None, outline=None, label=None):
        self.x, self.y = _f(x), _f(y)
        self.color = color
        self.marker = marker
        self.size = size
        self.values = None if values is None else _f(values)
        self.ramp = ramp
        self.norm = norm
        self.outline = outline
        self.label = label
        self.zorder = 3

    def bounds(self, xlog, ylog):
        return _xy_bounds(self.x, self.y, xlog, ylog)

    def points(self):
        return self.x, self.y

    def legend_entries(self):
        if not self.label:
            return []
        color = self.color
        if self.values is not None and self.ramp is not None:
            color = self.ramp.color_at(0.75)
        return [dict(kind="marker", label=self.label, color=color, marker=self.marker)]

    def draw(self, ctx):
        px, py = ctx.tx(self.x), ctx.ty(self.y)
        colors: Any = self.color
        if self.values is not None and self.ramp is not None:
            t = self.norm(self.values)
            colors = self.ramp.map(t[None, :], dither=False)[0]
            order = np.argsort(t, kind="stable")  # draw high values on top
            px, py, colors = px[order], py[order], colors[order]
        draw_markers(ctx, px, py, self.marker, self.size, colors, self.outline)


# ---------------------------------------------------------------------- bars
class Bars(Artist):
    def __init__(self, pos, length, width, base, colors, orientation="v", alpha=1.0,
                 outline="auto", label=None):
        self.pos = _f(pos)
        self.length = _f(length)
        n = len(self.pos)
        self.width = np.broadcast_to(_f(width), (n,)) if np.size(width) > 1 else np.full(n, width)
        self.base = np.broadcast_to(_f(base), (n,)) if np.size(base) > 1 else np.full(n, base)
        self.base = np.asarray(self.base, dtype=np.float64)
        self.width = np.asarray(self.width, dtype=np.float64)
        self.colors = colors  # list of RGB per bar
        self.orientation = orientation
        self.alpha = alpha
        self.outline = outline
        self.label = label
        self.zorder = 2
        if orientation == "v":
            self.sticky_y = tuple(np.unique(self.base))
        else:
            self.sticky_x = tuple(np.unique(self.base))

    def _extent(self):
        a = self.pos - self.width / 2
        b = self.pos + self.width / 2
        lo = np.minimum(self.base, self.base + self.length)
        hi = np.maximum(self.base, self.base + self.length)
        return a, b, lo, hi

    def bounds(self, xlog, ylog):
        a, b, lo, hi = self._extent()
        if self.orientation == "v":
            return _bounds4(a, b, lo, hi, xlog, ylog)
        return _bounds4(lo, hi, a, b, xlog, ylog)

    def points(self):
        a, b, lo, hi = self._extent()
        top = self.base + self.length
        if self.orientation == "v":
            return np.concatenate([self.pos, a, b]), np.concatenate([top, top, top])
        return np.concatenate([top, top, top]), np.concatenate([self.pos, a, b])

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="patch", label=self.label, color=self.colors[0], alpha=self.alpha,
                     outline=self.outline)]

    def draw(self, ctx):
        st = ctx.style
        a, b, _, _ = self._extent()
        tops = self.base + self.length
        for i in range(len(self.pos)):
            if not (np.isfinite(self.length[i]) and np.isfinite(self.pos[i])):
                continue
            color = self.colors[i % len(self.colors)]
            minimum = 1 if self.length[i] != 0 else 0
            if self.orientation == "v":
                x0, x1 = ctx.span(float(ctx.tx(a[i])), float(ctx.tx(b[i])), 1)
                y0, y1 = ctx.span(float(ctx.ty(self.base[i])), float(ctx.ty(tops[i])), minimum)
            else:
                x0, x1 = ctx.span(float(ctx.tx(self.base[i])), float(ctx.tx(tops[i])), minimum)
                y0, y1 = ctx.span(float(ctx.ty(a[i])), float(ctx.ty(b[i])), 1)
            draw_box(ctx.canvas, st, x0, y0, x1, y1, color, self.alpha, self.outline,
                     shading=st.bar_shading)


def _bounds4(x0, x1, y0, y1, xlog, ylog):
    xr = _finite_range(np.concatenate([_f(x0), _f(x1)]), xlog)
    yr = _finite_range(np.concatenate([_f(y0), _f(y1)]), ylog)
    if xr is None and yr is None:
        return None
    return (xr[0] if xr else None, xr[1] if xr else None,
            yr[0] if yr else None, yr[1] if yr else None)


def outline_color(style, color, outline):
    if outline is None:
        return None
    if isinstance(outline, str) and outline == "auto":
        return style.shade(color, -0.55)
    return style.color(outline)


def draw_box(canvas, style, x0, y0, x1, y1, color, alpha=1.0, outline="auto", shading=False):
    """Filled rectangle with optional pixel-art bevel shading and outline."""
    if x1 <= x0 or y1 <= y0:
        return
    canvas.fill_rect(x0, y0, x1, y1, color, alpha)
    w, h = x1 - x0, y1 - y0
    oc = outline_color(style, color, outline)
    if shading and w >= 4 and h >= 3 and alpha >= 1:
        hi = style.shade(color, 0.45)
        lo = style.shade(color, -0.35)
        canvas.fill_rect(x0 + 1, y0 + 1, x0 + 2, y1 - 1, hi)
        canvas.fill_rect(x0 + 1, y0 + 1, x1 - 1, y0 + 2, hi)
        canvas.fill_rect(x1 - 2, y0 + 2, x1 - 1, y1 - 1, lo)
    if oc is not None and w >= 3 and h >= 2:
        canvas.rect(x0, y0, x1, y1, oc)
    elif oc is not None and alpha < 1:
        canvas.rect(x0, y0, x1, y1, color)


# ---------------------------------------------------------------------- fills
class Polygon(Artist):
    """Filled polygon (used by ``fill`` and ``fill_between``)."""

    def __init__(self, x, y, color, alpha=1.0, edge=None, label=None, sticky_y=()):
        self.x, self.y = _f(x), _f(y)
        self.color = color
        self.alpha = alpha
        self.edge = edge
        self.label = label
        self.zorder = 1.8
        self.sticky_y = sticky_y

    def bounds(self, xlog, ylog):
        return _xy_bounds(self.x, self.y, xlog, ylog)

    def points(self):
        return self.x, self.y

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="patch", label=self.label, color=self.color, alpha=self.alpha,
                     outline=None)]

    def draw(self, ctx):
        px, py = ctx.tx(self.x), ctx.ty(self.y)
        ctx.canvas.fill_polygon(px, py, self.color, self.alpha)
        if self.edge is not None:
            ctx.canvas.polyline(np.append(px, px[:1]), np.append(py, py[:1]), self.edge)


class FillBetween(Artist):
    def __init__(self, x, y1, y2, color, alpha=1.0, edge=None, label=None):
        self.x = _f(x)
        n = len(self.x)
        self.y1 = np.broadcast_to(_f(y1), (n,)) if np.size(y1) > 1 else np.full(n, float(y1))
        self.y2 = np.broadcast_to(_f(y2), (n,)) if np.size(y2) > 1 else np.full(n, float(y2))
        self.color = color
        self.alpha = alpha
        self.edge = edge
        self.label = label
        self.zorder = 1.8
        sticky = []
        if np.size(y2) == 1:
            sticky.append(float(y2))
        if np.size(y1) == 1:
            sticky.append(float(y1))
        self.sticky_y = tuple(sticky)

    def bounds(self, xlog, ylog):
        return _xy_bounds(np.concatenate([self.x, self.x]),
                          np.concatenate([self.y1, self.y2]), xlog, ylog)

    def points(self):
        return np.concatenate([self.x, self.x]), np.concatenate([self.y1, self.y2])

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="patch", label=self.label, color=self.color, alpha=self.alpha,
                     outline=None)]

    def draw(self, ctx):
        ok = np.isfinite(self.x) & np.isfinite(self.y1) & np.isfinite(self.y2)
        # split into contiguous finite runs so NaNs produce gaps
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for r in runs:
            if r.size < 2:
                continue
            px = ctx.tx(self.x[r])
            p1, p2 = ctx.ty(self.y1[r]), ctx.ty(self.y2[r])
            ctx.canvas.fill_polygon(np.concatenate([px, px[::-1]]),
                                    np.concatenate([p1, p2[::-1]]), self.color, self.alpha)
            if self.edge is not None:
                ctx.canvas.polyline(px, p1, self.edge)
                ctx.canvas.polyline(px, p2, self.edge)


# ---------------------------------------------------------------------- error bars
class ErrorBars(Artist):
    def __init__(self, x, y, xerr, yerr, color, capsize=1, marker="o", markersize=1,
                 linestyle="none", label=None):
        self.x, self.y = _f(x), _f(y)
        self.xerr = _err(xerr, len(self.x))
        self.yerr = _err(yerr, len(self.x))
        self.color = color
        self.capsize = capsize
        self.marker = marker
        self.markersize = markersize
        self.linestyle = linestyle
        self.label = label
        self.zorder = 2.5

    def bounds(self, xlog, ylog):
        xs, ys = [self.x], [self.y]
        if self.xerr is not None:
            xs += [self.x - self.xerr[0], self.x + self.xerr[1]]
        if self.yerr is not None:
            ys += [self.y - self.yerr[0], self.y + self.yerr[1]]
        return _bounds4(np.concatenate(xs), np.concatenate(xs), np.concatenate(ys),
                        np.concatenate(ys), xlog, ylog)

    def points(self):
        return self.x, self.y

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="line", label=self.label, color=self.color, linestyle=self.linestyle,
                     width=1, marker=self.marker or "+")]

    def draw(self, ctx):
        cv, c, cap = ctx.canvas, self.color, self.capsize
        px, py = ctx.tx(self.x), ctx.ty(self.y)
        st = ctx.style
        if self.linestyle not in ("none", "", None):
            cv.polyline(px, py, c, 1, self.linestyle)
        nan = np.full_like(px, np.nan)
        if self.yerr is not None:
            ylo, yhi = ctx.ty(self.y - self.yerr[0]), ctx.ty(self.y + self.yerr[1])
            cv.polyline(np.column_stack([px, px, nan]).ravel(),
                        np.column_stack([ylo, yhi, nan]).ravel(), c)
            if cap:
                for yy in (ylo, yhi):
                    cv.polyline(np.column_stack([px - cap, px + cap, nan]).ravel(),
                                np.column_stack([yy, yy, nan]).ravel(), c)
        if self.xerr is not None:
            xlo, xhi = ctx.tx(self.x - self.xerr[0]), ctx.tx(self.x + self.xerr[1])
            cv.polyline(np.column_stack([xlo, xhi, nan]).ravel(),
                        np.column_stack([py, py, nan]).ravel(), c)
            if cap:
                for xx in (xlo, xhi):
                    cv.polyline(np.column_stack([xx, xx, nan]).ravel(),
                                np.column_stack([py - cap, py + cap, nan]).ravel(), c)
        if self.marker:
            outline = st.color(st.marker_outline) if st.marker_outline is not None else None
            draw_markers(ctx, px, py, self.marker, self.markersize, c, outline)


def _err(e, n):
    if e is None:
        return None
    e = np.asarray(e, dtype=np.float64)
    if e.ndim == 0:
        e = np.full(n, float(e))
    if e.ndim == 1:
        return np.abs(e), np.abs(e)
    return np.abs(e[0]), np.abs(e[1])


class Stems(Artist):
    def __init__(self, x, y, bottom, color, marker="o", markersize=1, label=None):
        self.x, self.y = _f(x), _f(y)
        self.bottom = float(bottom)
        self.color = color
        self.marker = marker
        self.markersize = markersize
        self.label = label
        self.sticky_y = (self.bottom,)

    def bounds(self, xlog, ylog):
        return _bounds4(self.x, self.x, np.append(self.y, self.bottom),
                        np.append(self.y, self.bottom), xlog, ylog)

    def points(self):
        return self.x, self.y

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="marker", label=self.label, color=self.color, marker=self.marker)]

    def draw(self, ctx):
        px, py = ctx.tx(self.x), ctx.ty(self.y)
        pb = np.full_like(py, float(ctx.ty(self.bottom)))
        nan = np.full_like(px, np.nan)
        ctx.canvas.polyline(np.column_stack([px, px, nan]).ravel(),
                            np.column_stack([pb, py, nan]).ravel(), self.color)
        if self.marker:
            st = ctx.style
            outline = st.color(st.marker_outline) if st.marker_outline is not None else None
            draw_markers(ctx, px, py, self.marker, self.markersize, self.color, outline)


# ---------------------------------------------------------------------- images
class Image(Artist):
    """Raster data drawn with nearest-neighbour lookup on (possibly irregular) cell edges."""

    zorder = 1

    def __init__(self, data, x_edges, y_edges, ramp: Ramp | None, norm: Norm | None,
                 dither=True, label=None):
        self.data = np.asarray(data)
        self.x_edges = _f(x_edges)
        self.y_edges = _f(y_edges)
        self.ramp = ramp
        self.norm = norm
        self.dither = dither
        self.label = label
        self.sticky_x = (self.x_edges[0], self.x_edges[-1])
        self.sticky_y = (self.y_edges[0], self.y_edges[-1])
        self.levels = None

    def bounds(self, xlog, ylog):
        return _bounds4(self.x_edges, self.x_edges, self.y_edges, self.y_edges, xlog, ylog)

    def _lookup(self, ctx):
        """Pixel region covered by the image and the data indices of each pixel."""
        x0, y0, x1, y1 = ctx.rect
        cols = np.arange(x0, x1)
        rows = np.arange(y0, y1)
        dx = ctx.inv_x(cols)
        dy = ctx.inv_y(rows)
        j, jok = _edge_index(self.x_edges, dx)
        i, iok = _edge_index(self.y_edges, dy)
        return x0, y0, i, iok, j, jok

    def sample(self, ctx):
        x0, y0, i, iok, j, jok = self._lookup(ctx)
        vals = self.data[i[:, None], j[None, :]]
        valid = iok[:, None] & jok[None, :]
        return x0, y0, vals, valid

    def draw(self, ctx):
        x0, y0, vals, valid = self.sample(ctx)
        if not valid.any():
            return
        if vals.ndim == 3:
            rgb = vals[..., :3]
            if vals.shape[-1] == 4:
                a = vals[..., 3]
                valid = valid & (a > (0.5 if a.dtype.kind == "f" else 127))
            st = ctx.style
            img = pixelate(rgb, palette=st.palette, bits=st.bits or 8, dither=self.dither)
        else:
            vals = vals.astype(np.float64)
            valid = valid & np.isfinite(vals)
            t = self.norm(vals)
            valid = valid & np.isfinite(t)
            img = self.ramp.map(t, self.dither, x0, y0)
        ctx.canvas.blit(img, x0, y0, mask=valid)


def _edge_index(edges: np.ndarray, v: np.ndarray):
    """Cell index for values ``v`` given monotonic cell ``edges``."""
    increasing = edges[-1] >= edges[0]
    e = edges if increasing else edges[::-1]
    idx = np.searchsorted(e, v, side="right") - 1
    ok = (idx >= 0) & (idx < len(e) - 1) & np.isfinite(v)
    idx = np.clip(idx, 0, len(e) - 2)
    if not increasing:
        idx = len(e) - 2 - idx
    return idx, ok


class FilledContour(Artist):
    """Filled contour bands rasterized at pixel resolution with bilinear interpolation."""

    zorder = 1

    def __init__(self, x, y, z, levels, ramp: Ramp, dither=True, label=None):
        self.x, self.y = _f(x), _f(y)
        self.z = np.asarray(z, dtype=np.float64)
        self.levels = np.asarray(levels, dtype=np.float64)
        self.ramp = ramp
        self.dither = dither
        self.label = label
        self.norm = Norm(self.levels[0], self.levels[-1])
        self.sticky_x = (self.x.min(), self.x.max())
        self.sticky_y = (self.y.min(), self.y.max())

    def bounds(self, xlog, ylog):
        return _bounds4(self.x, self.x, self.y, self.y, xlog, ylog)

    def band_value(self, k: int) -> float:
        n = len(self.levels) - 1
        return (k + 0.5) / n if n > 0 else 0.5

    def draw(self, ctx):
        x0, y0, x1, y1 = ctx.rect
        dx = ctx.inv_x(np.arange(x0, x1))
        dy = ctx.inv_y(np.arange(y0, y1))
        fj = _frac_index(self.x, dx)
        fi = _frac_index(self.y, dy)
        z = _bilinear(self.z, fi, fj)
        n = len(self.levels) - 1
        band = np.digitize(z, self.levels) - 1
        valid = np.isfinite(z) & (band >= 0) & (band < n)
        # values exactly on the top level belong to the last band
        top = np.isclose(z, self.levels[-1])
        band = np.where(top, n - 1, band)
        valid |= top
        t = (np.clip(band, 0, n - 1) + 0.5) / max(n, 1)
        img = self.ramp.map(t, self.dither, x0, y0)
        ctx.canvas.blit(img, x0, y0, mask=valid)


def _frac_index(coords: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fractional index of ``v`` along monotonic ``coords``; NaN outside."""
    if coords[0] > coords[-1]:
        coords = coords[::-1]
        f = np.interp(v, coords, np.arange(len(coords))[::-1].astype(float),
                      left=np.nan, right=np.nan)
    else:
        f = np.interp(v, coords, np.arange(len(coords), dtype=float), left=np.nan, right=np.nan)
    return f


def _bilinear(z: np.ndarray, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    ny, nx = z.shape
    ii = np.clip(np.nan_to_num(fi, nan=0.0), 0, ny - 1)
    jj = np.clip(np.nan_to_num(fj, nan=0.0), 0, nx - 1)
    i0 = np.minimum(np.floor(ii).astype(int), max(ny - 2, 0))
    j0 = np.minimum(np.floor(jj).astype(int), max(nx - 2, 0))
    i1 = np.minimum(i0 + 1, ny - 1)
    j1 = np.minimum(j0 + 1, nx - 1)
    ti = (ii - i0)[:, None]
    tj = (jj - j0)[None, :]
    z00 = z[i0[:, None], j0[None, :]]
    z01 = z[i0[:, None], j1[None, :]]
    z10 = z[i1[:, None], j0[None, :]]
    z11 = z[i1[:, None], j1[None, :]]
    out = (z00 * (1 - ti) * (1 - tj) + z01 * (1 - ti) * tj + z10 * ti * (1 - tj)
           + z11 * ti * tj)
    out[np.isnan(fi), :] = np.nan
    out[:, np.isnan(fj)] = np.nan
    return out


# marching squares: case -> list of (edge_a, edge_b)
_MS_TABLE = {
    1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)],
    7: [(3, 2)], 8: [(2, 3)], 9: [(0, 2)], 10: [(0, 1), (2, 3)], 11: [(1, 2)], 12: [(1, 3)],
    13: [(0, 1)], 14: [(3, 0)],
}


def marching_squares(x, y, z, level):
    """Contour line segments of ``z`` at ``level`` as NaN-separated x/y arrays."""
    z = np.asarray(z, dtype=np.float64)
    c0, c1 = z[:-1, :-1], z[:-1, 1:]
    c2, c3 = z[1:, 1:], z[1:, :-1]
    case = ((c0 > level) * 1 + (c1 > level) * 2 + (c2 > level) * 4 + (c3 > level) * 8)
    ok = np.isfinite(c0) & np.isfinite(c1) & np.isfinite(c2) & np.isfinite(c3)
    case = np.where(ok, case, 0)
    ii, jj = np.mgrid[0 : z.shape[0] - 1, 0 : z.shape[1] - 1]

    def interp(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (level - a) / (b - a)
        return np.clip(np.nan_to_num(t, nan=0.5), 0, 1)

    # edge points in fractional (i, j) index space
    edges = [
        (ii, jj + interp(c0, c1)),
        (ii + interp(c1, c2), jj + 1),
        (ii + 1, jj + interp(c3, c2)),
        (ii + interp(c0, c3), jj),
    ]
    xs, ys = [], []
    for k, pairs in _MS_TABLE.items():
        sel = case == k
        if not sel.any():
            continue
        for ea, eb in pairs:
            ia, ja = edges[ea][0][sel], edges[ea][1][sel]
            ib, jb = edges[eb][0][sel], edges[eb][1][sel]
            nan = np.full(ia.shape, np.nan)
            xs.append(np.column_stack([ja, jb, nan]).ravel())
            ys.append(np.column_stack([ia, ib, nan]).ravel())
    if not xs:
        return np.array([]), np.array([])
    fj = np.concatenate(xs)
    fi = np.concatenate(ys)
    gx = np.interp(fj, np.arange(len(x)), x)
    gy = np.interp(fi, np.arange(len(y)), y)
    gx[np.isnan(fj)] = np.nan
    gy[np.isnan(fi)] = np.nan
    return gx, gy


class ContourLines(Artist):
    def __init__(self, x, y, z, levels, colors, linestyles, width=1, label=None):
        self.x, self.y = _f(x), _f(y)
        self.levels = np.asarray(levels, dtype=np.float64)
        self.colors = colors
        self.linestyles = linestyles
        self.width = width
        self.label = label
        self.segments = [marching_squares(self.x, self.y, z, lv) for lv in self.levels]
        self.sticky_x = (self.x.min(), self.x.max())
        self.sticky_y = (self.y.min(), self.y.max())
        self.zorder = 2

    def bounds(self, xlog, ylog):
        return _bounds4(self.x, self.x, self.y, self.y, xlog, ylog)

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="line", label=self.label, color=self.colors[-1], linestyle="-",
                     width=self.width, marker=None)]

    def draw(self, ctx):
        for k, (sx, sy) in enumerate(self.segments):
            if sx.size == 0:
                continue
            ctx.canvas.polyline(ctx.tx(sx), ctx.ty(sy), self.colors[k], self.width,
                                self.linestyles[k])


# ---------------------------------------------------------------------- quiver
class Quiver(Artist):
    def __init__(self, x, y, u, v, colors, scale=None, pivot="tail", label=None):
        self.x, self.y, self.u, self.v = _f(x), _f(y), _f(u), _f(v)
        self.colors = colors  # RGB or (n, 3) array
        self.scale = scale
        self.pivot = pivot
        self.label = label
        self.zorder = 2.5

    def bounds(self, xlog, ylog):
        return _xy_bounds(self.x, self.y, xlog, ylog)

    def points(self):
        return self.x, self.y

    def legend_entries(self):
        if not self.label:
            return []
        c = self.colors if not isinstance(self.colors, np.ndarray) else tuple(self.colors[-1])
        return [dict(kind="line", label=self.label, color=c, linestyle="-", width=1,
                     marker=None)]

    def draw(self, ctx):
        px, py = ctx.tx(self.x), ctx.ty(self.y)
        mag = np.hypot(self.u, self.v)
        ok = np.isfinite(px) & np.isfinite(py) & np.isfinite(mag)
        if not ok.any():
            return
        scale = self.scale
        if scale is None:
            spacing = math.sqrt(ctx.width * ctx.height / max(ok.sum(), 1))
            scale = 0.9 * spacing / max(np.nanmax(mag[ok]), 1e-12)
        ddx, ddy = self.u * scale, -self.v * scale
        if self.pivot == "middle":
            sx, sy = px - ddx / 2, py - ddy / 2
        elif self.pivot == "tip":
            sx, sy = px - ddx, py - ddy
        else:
            sx, sy = px, py
        ex, ey = sx + ddx, sy + ddy
        per_point = isinstance(self.colors, np.ndarray) and self.colors.ndim == 2
        for k in np.flatnonzero(ok):
            c = tuple(int(v) for v in self.colors[k]) if per_point else self.colors
            draw_arrow(ctx.canvas, sx[k], sy[k], ex[k], ey[k], c)


def draw_arrow(canvas: Canvas, x0, y0, x1, y1, color, head: float | None = None):
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1.0:
        canvas.put([round(x0)], [round(y0)], color)
        return
    canvas.line(x0, y0, x1, y1, color)
    h = head if head is not None else min(3.0, max(1.5, length * 0.4))
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    for sgn in (-1, 1):
        hx = x1 - h * ux + sgn * h * 0.65 * uy
        hy = y1 - h * uy - sgn * h * 0.65 * ux
        canvas.line(x1, y1, hx, hy, color)


# ---------------------------------------------------------------------- pie
class Pie(Artist):
    def __init__(self, values, labels, colors, autopct=None, startangle=90.0, counterclock=True,
                 depth=0, outline=None, label_color=None):
        v = _f(values)
        if np.any(v < 0):
            raise ValueError("pie values must be non-negative")
        self.fracs = v / v.sum() if v.sum() > 0 else v
        self.labels = labels
        self.colors = colors
        self.autopct = autopct
        self.startangle = startangle
        self.counterclock = counterclock
        self.depth = depth
        self.outline = outline
        self.label_color = label_color

    def legend_entries(self):
        if not self.labels:
            return []
        return [dict(kind="patch", label=lb, color=self.colors[i % len(self.colors)], alpha=1.0,
                     outline="auto") for i, lb in enumerate(self.labels)]

    def wedge_index(self, dx, dy):
        ang = np.degrees(np.arctan2(-dy, dx))  # screen y grows downwards
        rel = (ang - self.startangle) * (1 if self.counterclock else -1)
        rel = np.mod(rel, 360.0) / 360.0
        cum = np.cumsum(self.fracs)
        return np.minimum(np.searchsorted(cum, rel, side="right"), len(self.fracs) - 1)

    def draw(self, ctx):
        st = ctx.style
        font = st.get_tick_font()
        x0, y0, x1, y1 = ctx.rect
        label_w = 0
        if self.labels:
            label_w = max(font.measure(st.text(s))[0] for s in self.labels) + 3
        label_h = font.line_height + 2 if self.labels else 0
        depth = self.depth
        if depth == "auto":
            depth = max(3, int(round((y1 - y0) * 0.05)))
        depth = int(depth or 0)
        tilt = 0.6 if depth > 0 else 1.0  # 3-D pies are viewed at an angle
        r = max(min((x1 - x0) / 2 - label_w,
                    ((y1 - y0 - depth) / 2 - label_h) / tilt) - 1, 3)
        cx = (x0 + x1 - 1) / 2
        cy = (y0 + y1 - 1 - depth) / 2
        n = int(math.ceil(r))
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        disc = xx * xx + (yy / tilt) ** 2 <= r * r + r * 0.8
        idx = self.wedge_index(xx, yy / tilt)
        cols = np.array([self.colors[i % len(self.colors)] for i in range(len(self.fracs))],
                        dtype=np.uint8)
        ix, iy = int(round(cx)) - n, int(round(cy)) - n
        edge = st.color(self.outline) if self.outline is not None else None
        # 3-D extrusion: a darker side wall below the front half of the pie
        if depth > 0:
            dark = np.array([st.shade(c, -0.45) for c in cols], dtype=np.uint8)
            size = 2 * n + 1
            side = np.full((size + depth, size), -1)
            front = disc & (yy >= -1)
            for d in range(depth, 0, -1):
                side[d : d + size][front] = idx[front]
            wall = side >= 0
            wall[:size] &= ~disc
            ctx.canvas.blit(dark[np.maximum(side, 0)], ix, iy, mask=wall)
            if edge is not None:
                left = np.roll(side, 1, axis=1)
                seams = wall & (side != left) & (left >= 0)
                below = np.zeros_like(wall)
                below[:-1] = wall[1:]
                outline = wall & (~below | ~np.roll(wall, 1, axis=1) | ~np.roll(wall, -1, axis=1))
                ctx.canvas.fill_mask(seams | outline, ix, iy, edge)
        ctx.canvas.blit(cols[idx], ix, iy, mask=disc)
        if edge is not None:
            # wedge boundaries and rim
            boundary = disc & ((idx != np.roll(idx, 1, axis=1)) | (idx != np.roll(idx, 1, axis=0)))
            if np.count_nonzero(self.fracs) > 1:
                ctx.canvas.fill_mask(boundary & disc, ix, iy, edge)
            rim = disc & ~(np.roll(disc, 1, 0) & np.roll(disc, -1, 0) & np.roll(disc, 1, 1)
                           & np.roll(disc, -1, 1))
            ctx.canvas.fill_mask(rim, ix, iy, edge)
        # labels
        start = 0.0
        for i, f in enumerate(self.fracs):
            mid = (start + f / 2) * 360.0
            start += f
            if f == 0:
                continue
            ang = math.radians(self.startangle + (mid if self.counterclock else -mid))
            ca, sn = math.cos(ang), -math.sin(ang)
            sa = sn * tilt
            if self.labels:
                lx = cx + r * ca + 3 * ca
                ly = cy + r * sa + 3 * sn + (depth if sn > 0 else 0)
                ha = "left" if ca > 0.15 else ("right" if ca < -0.15 else "center")
                va = "center" if abs(sn) < 0.5 else ("bottom" if sn < 0 else "top")
                ctx.canvas.text(round(lx), round(ly), st.text(self.labels[i]), font,
                                self.label_color or st.fg, ha=ha, va=va,
                                shadow=st.color(st.text_shadow) if st.text_shadow else None)
            if self.autopct:
                s = (self.autopct(f * 100) if callable(self.autopct)
                     else self.autopct % (f * 100))
                tw, th = font.measure(s)
                if f * 2 * math.pi * r * 0.6 > tw:
                    c = tuple(int(v) for v in cols[i])
                    txt = st.colors.snap((0, 0, 0)) if _bright(c) else st.colors.snap(
                        (255, 255, 255))
                    ctx.canvas.text(round(cx + 0.6 * r * ca), round(cy + 0.6 * r * sa), s, font,
                                    txt, ha="center", va="center")


def _bright(rgb) -> bool:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 140


# ---------------------------------------------------------------------- statistics
def box_stats(data, whis=1.5):
    d = _f(data)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    iqr = q3 - q1
    lo_lim, hi_lim = q1 - whis * iqr, q3 + whis * iqr
    inside = d[(d >= lo_lim) & (d <= hi_lim)]
    wlo = inside.min() if inside.size else q1
    whi = inside.max() if inside.size else q3
    fliers = d[(d < wlo) | (d > whi)]
    return dict(q1=q1, med=med, q3=q3, wlo=wlo, whi=whi, fliers=fliers, mean=d.mean())


class BoxPlot(Artist):
    def __init__(self, datasets, positions, widths, colors, showfliers=True, label=None):
        self.stats = [box_stats(d) for d in datasets]
        self.positions = _f(positions)
        self.widths = np.broadcast_to(_f(widths), self.positions.shape)
        self.colors = colors
        self.showfliers = showfliers
        self.label = label

    def bounds(self, xlog, ylog):
        ys = []
        for s in self.stats:
            if s is None:
                continue
            ys += [s["wlo"], s["whi"]]
            if self.showfliers and s["fliers"].size:
                ys += [s["fliers"].min(), s["fliers"].max()]
        p, w = self.positions, self.widths
        return _bounds4(p - w / 2, p + w / 2, np.array(ys), np.array(ys), xlog, ylog)

    def points(self):
        xs, ys = [], []
        for p, s in zip(self.positions, self.stats):
            if s is not None:
                xs += [p, p]
                ys += [s["wlo"], s["whi"]]
        return np.array(xs), np.array(ys)

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="patch", label=self.label, color=self.colors[0], alpha=1.0,
                     outline="auto")]

    def draw(self, ctx):
        st = ctx.style
        fg = st.fg
        cv = ctx.canvas
        for k, (p, w, s) in enumerate(zip(self.positions, self.widths, self.stats)):
            if s is None:
                continue
            color = self.colors[k % len(self.colors)]
            edge = st.shade(color, -0.55) if st.bar_shading else fg
            x0, x1 = ctx.span(float(ctx.tx(p - w / 2)), float(ctx.tx(p + w / 2)), 3)
            ya, yb = ctx.span(float(ctx.ty(s["q1"])), float(ctx.ty(s["q3"])), 1)
            xc = float(ctx.tx(p))
            cap_w = max((x1 - x0) // 4, 1)
            for v_end, v_box in ((s["whi"], s["q3"]), (s["wlo"], s["q1"])):
                ye, yq = float(ctx.ty(v_end)), float(ctx.ty(v_box))
                cv.line(xc, ye, xc, yq, fg)
                cv.line(xc - cap_w, ye, xc + cap_w, ye, fg)
            draw_box(cv, st, x0, ya - 1, x1, yb + 1, color, 1.0, "auto" if st.bar_shading else fg,
                     shading=st.bar_shading)
            ym = float(ctx.ty(s["med"]))
            med_c = st.colors.snap((255, 255, 255)) if not _bright(color) else st.colors.snap(
                (0, 0, 0))
            cv.line(x0 + 1, ym, x1 - 2, ym, med_c if edge != med_c else fg)
            if self.showfliers and s["fliers"].size:
                outline = st.color(st.marker_outline) if st.marker_outline is not None else None
                draw_markers(ctx, np.full(s["fliers"].shape, xc), ctx.ty(s["fliers"]), ".", 1,
                             color, outline)


def gaussian_kde(data: np.ndarray, points: np.ndarray, bw: float | None = None) -> np.ndarray:
    """Gaussian kernel density estimate (Scott's rule bandwidth)."""
    d = data[np.isfinite(data)]
    if d.size > 2000:
        d = np.random.default_rng(0).choice(d, 2000, replace=False)
    if d.size == 0:
        return np.zeros_like(points)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    if bw is None:
        bw = sd * d.size ** (-1 / 5) if sd > 0 else 1.0
    z = (points[:, None] - d[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (d.size * bw * math.sqrt(2 * math.pi))


class Violins(Artist):
    def __init__(self, datasets, positions, widths, colors, show_box=True, label=None):
        self.data = [_f(d) for d in datasets]
        self.positions = _f(positions)
        self.widths = np.broadcast_to(_f(widths), self.positions.shape)
        self.colors = colors
        self.show_box = show_box
        self.label = label

    def bounds(self, xlog, ylog):
        ys = np.concatenate([d[np.isfinite(d)] for d in self.data]) if self.data else np.array([])
        p, w = self.positions, self.widths
        return _bounds4(p - w / 2, p + w / 2, ys, ys, xlog, ylog)

    def points(self):
        xs, ys = [], []
        for p, d in zip(self.positions, self.data):
            d = d[np.isfinite(d)]
            if d.size:
                xs += [p, p]
                ys += [d.min(), d.max()]
        return np.array(xs), np.array(ys)

    def legend_entries(self):
        if not self.label:
            return []
        return [dict(kind="patch", label=self.label, color=self.colors[0], alpha=1.0,
                     outline="auto")]

    def draw(self, ctx):
        st = ctx.style
        cv = ctx.canvas
        x0, y0, x1, y1 = ctx.rect
        for k, (p, w, d) in enumerate(zip(self.positions, self.widths, self.data)):
            d = d[np.isfinite(d)]
            if d.size == 0:
                continue
            color = self.colors[k % len(self.colors)]
            edge = st.shade(color, -0.55)
            ra, rb = ctx.span(float(ctx.ty(d.max())), float(ctx.ty(d.min())), 1)
            ra, rb = max(ra, y0), min(rb + 1, y1)
            if rb <= ra:
                continue
            rows = np.arange(ra, rb)
            dens = gaussian_kde(d, ctx.inv_y(rows))
            half = (float(ctx.tx(p + w / 2)) - float(ctx.tx(p - w / 2))) / 2
            hw = np.round(dens / max(dens.max(), 1e-300) * half).astype(int)
            xc = int(round(float(ctx.tx(p))))
            mask = np.zeros((len(rows), 2 * int(math.ceil(half)) + 1), dtype=bool)
            cols = np.arange(mask.shape[1]) - mask.shape[1] // 2
            mask[:] = np.abs(cols)[None, :] <= hw[:, None]
            ox = xc - mask.shape[1] // 2
            cv.fill_mask(mask, ox, ra, color)
            inner = mask.copy()
            inner[:, 1:-1] = mask[:, :-2] & mask[:, 2:] & mask[:, 1:-1]
            inner[:, [0, -1]] = False
            inner[1:-1] &= mask[:-2, :] & mask[2:, :]
            inner[[0, -1], :] = False
            cv.fill_mask(mask & ~inner, ox, ra, edge)
            if self.show_box:
                s = box_stats(d)
                cv.line(xc, ctx.ty(s["q1"]), xc, ctx.ty(s["q3"]), edge, width=3 if half > 4 else 1)
                cv.put([xc], [int(round(float(ctx.ty(s["med"]))))],
                       st.colors.snap((255, 255, 255)))


# ---------------------------------------------------------------------- text
class Text(Artist):
    zorder = 5

    def __init__(self, x, y, s, color, ha="left", va="baseline", coords="data", font=None,
                 box=False, shadow=None):
        self.x, self.y, self.s = x, y, s
        self.color = color
        self.ha, self.va = ha, va
        self.coords = coords
        self.font = font
        self.box = box
        self.shadow = shadow

    def anchor(self, ctx):
        if self.coords == "axes":
            x0, y0, x1, y1 = ctx.rect
            return x0 + self.x * (x1 - x0 - 1), y1 - 1 - self.y * (y1 - y0 - 1)
        if self.coords == "pixels":
            return ctx.rect[0] + self.x, ctx.rect[1] + self.y
        return float(ctx.tx(self.x)), float(ctx.ty(self.y))

    def draw(self, ctx):
        st = ctx.style
        font = self.font or st.get_tick_font()
        x, y = self.anchor(ctx)
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        s = st.text(self.s)
        if self.box:
            tw, th = font.measure(s)
            bx = {"left": x, "center": x - tw // 2, "right": x - tw + 1}[self.ha]
            cap = th - font.descent
            by = {"top": y, "center": y - cap // 2, "baseline": y - cap + 1,
                  "bottom": y - th + 1}[self.va]
            bx, by = int(round(bx)), int(round(by))
            fill = st.legend_fill or st.panel_fill or st.background
            fill = fill[0] if isinstance(fill, tuple) and isinstance(fill[0], str) else fill
            ctx.canvas.fill_rect(bx - 2, by - 2, bx + tw + 2, by + th + 1, st.color(fill))
            ctx.canvas.rect(bx - 2, by - 2, bx + tw + 2, by + th + 1, st.fg)
        shadow = self.shadow
        if shadow is None and st.text_shadow is not None:
            shadow = st.color(st.text_shadow)
        ctx.canvas.text(int(round(x)), int(round(y)), s, font, self.color, self.ha, self.va,
                        shadow=shadow)


class Annotation(Text):
    def __init__(self, s, xy, xytext, color, arrow=True, coords="data", **kw):
        super().__init__(xytext[0], xytext[1], s, color, coords=coords, **kw)
        self.xy = xy
        self.arrow = arrow

    def bounds(self, xlog, ylog):
        if self.coords != "data":
            return None
        return _xy_bounds([self.xy[0]], [self.xy[1]], xlog, ylog)

    def draw(self, ctx):
        st = ctx.style
        font = self.font or st.get_tick_font()
        tx, ty = self.anchor(ctx)
        px, py = float(ctx.tx(self.xy[0])), float(ctx.ty(self.xy[1]))
        if self.arrow and all(np.isfinite([tx, ty, px, py])):
            tw, th = font.measure(st.text(self.s))
            bx0 = {"left": tx, "center": tx - tw / 2, "right": tx - tw}[self.ha]
            cap = th - font.descent
            by0 = {"top": ty, "center": ty - cap / 2, "baseline": ty - cap,
                   "bottom": ty - th}[self.va]
            sx = min(max(px, bx0 - 2), bx0 + tw + 1)
            sy = min(max(py, by0 - 2), by0 + th + 1)
            if math.hypot(px - sx, py - sy) > 3:
                # stop just short of the target point
                d = math.hypot(px - sx, py - sy)
                ex, ey = px - (px - sx) / d * 2, py - (py - sy) / d * 2
                draw_arrow(ctx.canvas, sx, sy, ex, ey, self.color)
        super().draw(ctx)


def legend_handle(canvas: Canvas, style, entry: dict, x: int, y: int, w: int, h: int):
    """Draw a legend key glyph in the box ``[x, x+w) x [y, y+h)``."""
    kind = entry["kind"]
    color = entry["color"]
    cy = y + h // 2
    if kind == "line":
        ls = entry.get("linestyle") or "-"
        if ls not in ("none", ""):
            pattern = dash_pattern(ls)
            if pattern is not None and len(pattern) > w // 2:
                ls = (1, 1, 0)
            canvas.polyline([x, x + w - 1], [cy, cy], color, entry.get("width", 1), ls)
        if entry.get("marker"):
            m = marker_mask(entry["marker"], 1)
            canvas.fill_mask(m, x + w // 2 - m.shape[1] // 2, cy - m.shape[0] // 2, color)
    elif kind == "marker":
        m = marker_mask(entry["marker"], 1)
        canvas.fill_mask(m, x + w // 2 - m.shape[1] // 2, cy - m.shape[0] // 2, color)
    else:
        bh = min(h, 5)
        draw_box(canvas, style, x + 1, cy - bh // 2, x + w - 1, cy - bh // 2 + bh, color,
                 entry.get("alpha", 1.0), entry.get("outline", "auto"), shading=False)
