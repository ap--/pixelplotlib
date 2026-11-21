"""Axes: data coordinates, layout, decorations and the plotting API."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import artists as A
from .artists import RenderContext, Scale, make_norm
from .canvas import Canvas
from .colormaps import CmapLike, Ramp
from .ticks import format_ticks, log_ticks, nice_ticks

if TYPE_CHECKING:  # pragma: no cover
    from .figure import Figure

_TICK_GAP = 2  # pixels between a y tick mark and its label


def _overlaps(a, b) -> bool:
    """Whether two ``(x0, y0, x1, y1)`` boxes (exclusive ends) intersect."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


_LOCS = {
    "upper right": (1, 0), "upper left": (0, 0), "lower left": (0, 1), "lower right": (1, 1),
    "upper center": (0.5, 0), "lower center": (0.5, 1), "center left": (0, 0.5),
    "center right": (1, 0.5), "center": (0.5, 0.5),
}

_FMT_COLORS = {"r": "red", "g": "green", "b": "blue", "c": "cyan", "m": "magenta",
               "y": "yellow", "k": "black", "w": "white"}
_FMT_MARKERS = "osDdv^+x*.p"


def _parse_fmt(fmt: str):
    color = marker = linestyle = None
    s = fmt
    for ls in ("--", "-.", "-", ":"):
        if ls in s:
            linestyle = ls
            s = s.replace(ls, "", 1)
            break
    for ch in s:
        if ch in _FMT_COLORS:
            color = _FMT_COLORS[ch]
        elif ch in _FMT_MARKERS:
            marker = "D" if ch == "d" else ch
        else:
            raise ValueError(f"unrecognized format string {fmt!r}")
    if marker and linestyle is None:
        linestyle = "none"
    return color, marker, linestyle


def _is_str_seq(x) -> bool:
    if isinstance(x, (str, bytes)) or not hasattr(x, "__len__") or len(x) == 0:
        return False
    first = x[0]
    return isinstance(first, str)


class Colorbar:
    def __init__(self, mappable, label=None, ticks=None):
        self.mappable = mappable
        self.label = label
        self.ticks = ticks

    def tick_values(self, max_ticks: int):
        m = self.mappable
        if self.ticks is not None:
            vals = np.asarray(self.ticks, dtype=np.float64)
        elif getattr(m, "levels", None) is not None:
            vals = np.asarray(m.levels, dtype=np.float64)
            stride = max(1, math.ceil(len(vals) / max_ticks))
            vals = vals[::stride]
        elif m.norm.kind == "log":
            vals = log_ticks(m.norm.vmin, m.norm.vmax, max_ticks)
        else:
            vals, _ = nice_ticks(m.norm.vmin, m.norm.vmax, max_ticks)
        lo, hi = sorted((m.norm.vmin, m.norm.vmax))
        tol = (hi - lo) * 1e-9
        vals = vals[(vals >= lo - tol) & (vals <= hi + tol)]
        return vals, format_ticks(vals, log=m.norm.kind == "log")


class Axes:
    """A single plot panel. Create via :meth:`Figure.add_subplot` or :func:`subplots`."""

    def __init__(self, figure: Figure, spec):
        self.figure = figure
        self._spec = spec
        self.artists: list[A.Artist] = []
        self._title = self._xlabel = self._ylabel = None
        self._xlim: list[float | None] = [None, None]
        self._ylim: list[float | None] = [None, None]
        self._xscale = Scale("linear")
        self._yscale = Scale("linear")
        self._xticks = self._yticks = None
        self._xticklabels = self._yticklabels = None
        self._grid: str | None = None
        self._legend: dict | None = None
        self._colorbar: Colorbar | None = None
        self._axis_on = True
        self._aspect = "auto"
        self._margins = [0.05, 0.05]
        self._series = 0
        self._xcats: dict[str, int] = {}
        self._ycats: dict[str, int] = {}
        self._invert = [False, False]
        self.last_rect: tuple[int, int, int, int] | None = None

    # ------------------------------------------------------------------ basics
    @property
    def style(self):
        return self.figure.style

    def _cycle(self) -> dict:
        st = self.style
        i = self._series
        self._series += 1
        return dict(
            color=st.series_color(i),
            linestyle=st.linestyles[i % len(st.linestyles)],
            marker=st.markers[i % len(st.markers)],
            alpha=st.fill_alphas[i % len(st.fill_alphas)],
        )

    def _color(self, c, default):
        return self.style.color(c) if c is not None else default

    def _colors(self, c, n, default):
        """Resolve ``c`` (None, one color or one per item) to a list of RGB tuples."""
        if c is None:
            return [default]
        if isinstance(c, (list, tuple)) and len(c) == n and n > 0 and not (
            n in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in c)
        ):
            return [self.style.color(v) for v in c]
        return [self.style.color(c)]

    def _add(self, artist: A.Artist) -> A.Artist:
        self.artists.append(artist)
        return artist

    def _cat(self, values, axis: str):
        """Map categorical (string) values to integer positions."""
        if not _is_str_seq(values):
            return np.asarray(values, dtype=np.float64)
        cats = self._xcats if axis == "x" else self._ycats
        out = []
        for v in values:
            if v not in cats:
                cats[v] = len(cats)
            out.append(cats[v])
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------------ setters
    def set_title(self, title: str | None):
        self._title = title
        return self

    def set_xlabel(self, label: str | None):
        self._xlabel = label
        return self

    def set_ylabel(self, label: str | None):
        self._ylabel = label
        return self

    def set_xlim(self, left=None, right=None):
        if isinstance(left, (tuple, list)):
            left, right = left
        self._xlim = [left, right]
        return self

    def set_ylim(self, bottom=None, top=None):
        if isinstance(bottom, (tuple, list)):
            bottom, top = bottom
        self._ylim = [bottom, top]
        return self

    def get_xlim(self) -> tuple[float, float]:
        return self._limits()[0]

    def get_ylim(self) -> tuple[float, float]:
        return self._limits()[1]

    def set_xscale(self, scale: str):
        self._xscale = Scale(scale)
        return self

    def set_yscale(self, scale: str):
        self._yscale = Scale(scale)
        return self

    def set_xticks(self, ticks, labels: Sequence[str] | None = None):
        self._xticks = None if ticks is None else np.asarray(ticks, dtype=np.float64)
        self._xticklabels = None if labels is None else [str(s) for s in labels]
        return self

    def set_yticks(self, ticks, labels: Sequence[str] | None = None):
        self._yticks = None if ticks is None else np.asarray(ticks, dtype=np.float64)
        self._yticklabels = None if labels is None else [str(s) for s in labels]
        return self

    def set_xticklabels(self, labels):
        self._xticklabels = [str(s) for s in labels]
        return self

    def set_yticklabels(self, labels):
        self._yticklabels = [str(s) for s in labels]
        return self

    def set_aspect(self, aspect: str | float):
        """``"auto"`` (fill the panel) or ``"equal"`` / a number (data units y per x)."""
        self._aspect = aspect
        return self

    def margins(self, x: float | None = None, y: float | None = None):
        """Fractional padding added around the data when autoscaling."""
        if y is None:
            y = x
        if x is not None:
            self._margins[0] = x
        if y is not None:
            self._margins[1] = y
        return self

    def invert_xaxis(self):
        self._invert[0] = not self._invert[0]
        return self

    def invert_yaxis(self):
        self._invert[1] = not self._invert[1]
        return self

    def grid(self, visible: bool = True, axis: str = "both"):
        self._grid = axis if visible else None
        return self

    def axis(self, state: str | bool = "on"):
        """``axis("off")`` hides frame, ticks and labels."""
        self._axis_on = state not in ("off", False)
        return self

    def set_axis_off(self):
        return self.axis("off")

    def legend(self, loc: str = "best", title: str | None = None):
        """Show a legend of all labelled artists."""
        if loc != "best" and loc not in _LOCS:
            raise ValueError(f"unknown legend loc {loc!r}; choose from {['best', *_LOCS]}")
        self._legend = dict(loc=loc, title=title)
        return self

    def colorbar(self, mappable=None, label: str | None = None, ticks=None):
        """Attach a colorbar for ``mappable`` (defaults to the last colormapped artist)."""
        if mappable is None:
            mappable = next((a for a in reversed(self.artists)
                             if getattr(a, "ramp", None) is not None
                             and getattr(a, "norm", None) is not None), None)
            if mappable is None:
                raise ValueError("no colormapped artist to attach a colorbar to")
        self._colorbar = Colorbar(mappable, label, ticks)
        return self._colorbar

    def set(self, **kwargs):
        """Set several properties at once, e.g. ``ax.set(title=..., xlabel=..., xlim=...)``."""
        for key, value in kwargs.items():
            setter = getattr(self, f"set_{key}", None)
            if setter is None:
                raise AttributeError(f"unknown property {key!r}")
            if isinstance(value, (tuple, list)) and key in ("xlim", "ylim"):
                setter(*value)
            else:
                setter(value)
        return self

    # ------------------------------------------------------------------ plotting API
    def plot(self, *args, color=None, linewidth=None, lw=None, linestyle=None, ls=None,
             marker=None, markersize=1, ms=None, label=None):
        """Line plot: ``plot(y)``, ``plot(x, y)`` or ``plot(x, y, fmt)`` (e.g. ``"r--o"``).

        A 2-D ``y`` draws one line per column.
        """
        fmt = None
        if args and isinstance(args[-1], str):
            fmt = args[-1]
            args = args[:-1]
        if len(args) == 1:
            y = np.asarray(args[0], dtype=np.float64)
            x = np.arange(y.shape[0], dtype=np.float64)
        elif len(args) == 2:
            x, y = self._cat(args[0], "x"), np.asarray(args[1], dtype=np.float64)
        else:
            raise TypeError("plot() takes y, (x, y) or (x, y, fmt)")
        fcolor = fmarker = fls = None
        if fmt:
            fcolor, fmarker, fls = _parse_fmt(fmt)
        lines = []
        ys = y.T if y.ndim == 2 else [y]
        labels = label if isinstance(label, (list, tuple)) else [label] * len(ys)
        for k, yy in enumerate(ys):
            cyc = self._cycle()
            lines.append(self._add(A.Line(
                x, yy,
                color=self._color(color or fcolor, cyc["color"]),
                width=linewidth or lw or self.style.line_width,
                linestyle=linestyle or ls or fls or cyc["linestyle"],
                marker=marker or fmarker,
                markersize=ms or markersize,
                label=labels[k],
            )))
        return lines if len(lines) > 1 else lines[0]

    def step(self, x, y, where: str = "pre", **kwargs):
        """Staircase plot; ``where`` is ``"pre"``, ``"post"`` or ``"mid"``."""
        x = self._cat(x, "x")
        y = np.asarray(y, dtype=np.float64)
        if len(x) < 2:
            return self.plot(x, y, **kwargs)
        if where == "pre":
            sx, sy = np.repeat(x, 2)[:-1], np.repeat(y, 2)[1:]
        elif where == "post":
            sx, sy = np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]
        elif where == "mid":
            mids = (x[:-1] + x[1:]) / 2
            sx = np.concatenate([[x[0]], np.repeat(mids, 2), [x[-1]]])
            sy = np.repeat(y, 2)
        else:
            raise ValueError("where must be 'pre', 'post' or 'mid'")
        return self.plot(sx, sy, **kwargs)

    def scatter(self, x, y, c=None, color=None, marker=None, size=1, s=None, cmap=None,
                vmin=None, vmax=None, norm=None, outline=None, label=None):
        """Scatter plot with pixel sprites. ``c`` may be values to colormap."""
        cyc = self._cycle()
        x = self._cat(x, "x")
        y = self._cat(y, "y")
        values = ramp = nrm = None
        if c is not None and np.ndim(c) == 1 and len(c) == len(x) and not isinstance(c[0], str):
            values = np.asarray(c, dtype=np.float64)
            ramp = self.style.ramp(cmap)
            nrm = make_norm(values, vmin, vmax, norm)
        elif c is not None:
            color = c
        st = self.style
        if outline is None and st.marker_outline is not None:
            outline = st.marker_outline
        return self._add(A.Scatter(
            x, y, self._color(color, cyc["color"]), marker=marker or cyc["marker"],
            size=s or size, values=values, ramp=ramp, norm=nrm,
            outline=st.color(outline) if outline not in (None, False) else None, label=label,
        ))

    def bar(self, x, height, width=0.8, bottom=0.0, color=None, alpha=None, edgecolor="auto",
            label=None):
        """Vertical bars. ``x`` may be category names."""
        cyc = self._cycle()
        x = self._cat(x, "x")
        h = np.broadcast_to(np.asarray(height, dtype=np.float64), x.shape)
        return self._add(A.Bars(
            x, h, width, bottom, self._colors(color, len(x), cyc["color"]), "v",
            alpha=cyc["alpha"] if alpha is None else alpha,
            outline=self._outline(edgecolor), label=label,
        ))

    def barh(self, y, width, height=0.8, left=0.0, color=None, alpha=None, edgecolor="auto",
             label=None):
        """Horizontal bars. ``y`` may be category names."""
        cyc = self._cycle()
        y = self._cat(y, "y")
        w = np.broadcast_to(np.asarray(width, dtype=np.float64), y.shape)
        return self._add(A.Bars(
            y, w, height, left, self._colors(color, len(y), cyc["color"]), "h",
            alpha=cyc["alpha"] if alpha is None else alpha,
            outline=self._outline(edgecolor), label=label,
        ))

    def _outline(self, edgecolor):
        if edgecolor == "auto":
            return self.style.bar_outline
        return edgecolor

    def hist(self, data, bins=10, range=None, density=False, cumulative=False, color=None,
             alpha=None, histtype="bar", edgecolor="auto", label=None, weights=None):
        """Histogram. Returns ``(counts, edges, artist)``."""
        d = np.asarray(data, dtype=np.float64).ravel()
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).ravel()[np.isfinite(d)]
        d = d[np.isfinite(d)]
        if self._xscale.log and isinstance(bins, int):
            pos = d[d > 0]
            lo, hi = range or (pos.min(), pos.max())
            bins = np.logspace(math.log10(lo), math.log10(hi), bins + 1)
        counts, edges = np.histogram(d, bins=bins, range=range, density=density,
                                     weights=weights)
        if cumulative:
            counts = np.cumsum(counts)
            if density:
                counts = counts / counts[-1]
        if histtype == "step":
            x = np.repeat(edges, 2)
            y = np.concatenate([[0], np.repeat(counts, 2), [0]])
            art = self.plot(x, y, color=color, label=label)
            art.sticky_y = (0.0,)
            return counts, edges, art
        cyc = self._cycle()
        art = self._add(A.Bars(
            (edges[:-1] + edges[1:]) / 2, counts, np.diff(edges), 0.0,
            self._colors(color, len(counts), cyc["color"]), "v",
            alpha=cyc["alpha"] if alpha is None else alpha, outline=self._outline(edgecolor),
            label=label,
        ))
        return counts, edges, art

    def hist2d(self, x, y, bins=20, range=None, density=False, cmap=None, norm=None,
               vmin=None, vmax=None, dither=None):
        """2-D histogram rendered as a heatmap. Empty bins are left transparent."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        h, xe, ye = np.histogram2d(x[ok], y[ok], bins=bins, range=range, density=density)
        h = h.T.astype(np.float64)
        h[h == 0] = np.nan
        art = self.pcolormesh(xe, ye, h, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                              dither=dither)
        return h, xe, ye, art

    def fill_between(self, x, y1, y2=0.0, color=None, alpha=None, edgecolor=None, label=None):
        """Fill the area between two curves (``alpha`` is emulated with dithering)."""
        cyc = self._cycle()
        c = self._color(color, cyc["color"])
        if alpha is None:
            alpha = min(cyc["alpha"], 0.5) if len(self.style.fill_alphas) > 1 else 0.5
        edge = self.style.color(edgecolor) if edgecolor not in (None, "none") else None
        return self._add(A.FillBetween(self._cat(x, "x"), y1, y2, c, alpha, edge, label))

    def fill(self, x, y, color=None, alpha=1.0, edgecolor=None, label=None):
        """Fill a polygon."""
        cyc = self._cycle()
        edge = self.style.color(edgecolor) if edgecolor not in (None, "none") else None
        return self._add(A.Polygon(x, y, self._color(color, cyc["color"]), alpha, edge, label))

    def errorbar(self, x, y, yerr=None, xerr=None, color=None, marker="o", markersize=1,
                 capsize=1, linestyle="none", label=None):
        """Points with error bars. Errors can be scalars, arrays or ``(lower, upper)`` pairs."""
        cyc = self._cycle()
        return self._add(A.ErrorBars(self._cat(x, "x"), y, xerr, yerr,
                                     self._color(color, cyc["color"]), capsize, marker,
                                     markersize, linestyle, label))

    def stem(self, x, y=None, bottom=0.0, color=None, marker="o", label=None):
        """Stem plot (vertical lines from ``bottom`` to each value)."""
        if y is None:
            y, x = x, np.arange(len(x))
        cyc = self._cycle()
        return self._add(A.Stems(self._cat(x, "x"), y, bottom, self._color(color, cyc["color"]),
                                 marker, 1, label))

    def axhline(self, y=0.0, color=None, linestyle="-", linewidth=1, label=None):
        return self._add(A.RefLine(y, "h", self._color(color, self.style.fg), linestyle,
                                   linewidth, label))

    def axvline(self, x=0.0, color=None, linestyle="-", linewidth=1, label=None):
        return self._add(A.RefLine(x, "v", self._color(color, self.style.fg), linestyle,
                                   linewidth, label))

    def axhspan(self, ymin, ymax, color=None, alpha=0.5, label=None):
        return self._add(A.Span(ymin, ymax, "h", self._color(color, self.style.fg), alpha,
                                label))

    def axvspan(self, xmin, xmax, color=None, alpha=0.5, label=None):
        return self._add(A.Span(xmin, xmax, "v", self._color(color, self.style.fg), alpha,
                                label))

    def imshow(self, data, cmap: CmapLike | None = None, vmin=None, vmax=None, norm=None,
               extent=None, origin: str = "upper", aspect: str | None = None, dither=None):
        """Display a 2-D array (colormapped) or an RGB(A) image (pixelated to the palette).

        ``extent`` is ``(left, right, bottom, top)``. ``dither=False`` gives posterized bands.
        """
        arr = np.asarray(data)
        if arr.ndim not in (2, 3):
            raise ValueError("imshow expects a 2-D array or an (H, W, 3|4) image")
        h, w = arr.shape[:2]
        if extent is None:
            extent = (-0.5, w - 0.5, h - 0.5, -0.5) if origin == "upper" else (
                -0.5, w - 0.5, -0.5, h - 0.5)
        left, right, bottom, top = extent
        x_edges = np.linspace(left, right, w + 1)
        # row 0 is drawn at ``top`` for origin="upper" and at ``bottom`` for "lower"
        y_edges = np.linspace(top, bottom, h + 1) if origin == "upper" else np.linspace(
            bottom, top, h + 1)
        if aspect is not None:
            self.set_aspect(aspect)
        if left > right:
            self._invert[0] = True
        if bottom > top:
            self._invert[1] = True
        return self._add_image(arr, x_edges, y_edges, cmap, vmin, vmax, norm, dither)

    def pcolormesh(self, *args, cmap=None, vmin=None, vmax=None, norm=None, dither=None):
        """``pcolormesh(C)`` or ``pcolormesh(x_edges, y_edges, C)`` for (ir)regular grids."""
        if len(args) == 1:
            c = np.asarray(args[0])
            xe = np.arange(c.shape[1] + 1, dtype=np.float64)
            ye = np.arange(c.shape[0] + 1, dtype=np.float64)
        elif len(args) == 3:
            xe, ye, c = (np.asarray(a) for a in args)
            if xe.ndim == 2:
                xe = xe[0]
            if ye.ndim == 2:
                ye = ye[:, 0]
            if len(xe) == c.shape[1]:
                xe = _centers_to_edges(xe)
            if len(ye) == c.shape[0]:
                ye = _centers_to_edges(ye)
        else:
            raise TypeError("pcolormesh takes C or (x, y, C)")
        return self._add_image(c, xe, ye, cmap, vmin, vmax, norm, dither)

    def _add_image(self, arr, x_edges, y_edges, cmap, vmin, vmax, norm, dither):
        dither = self.style.dither if dither is None else dither
        if arr.ndim == 3:
            return self._add(A.Image(arr, x_edges, y_edges, None, None, dither))
        ramp = self.style.ramp(cmap)
        nrm = make_norm(arr, vmin, vmax, norm)
        return self._add(A.Image(arr, x_edges, y_edges, ramp, nrm, dither))

    def _grid_args(self, args):
        if len(args) == 1:
            z = np.asarray(args[0], dtype=np.float64)
            x = np.arange(z.shape[1], dtype=np.float64)
            y = np.arange(z.shape[0], dtype=np.float64)
        elif len(args) == 3:
            x, y, z = (np.asarray(a, dtype=np.float64) for a in args)
            if x.ndim == 2:
                x = x[0]
            if y.ndim == 2:
                y = y[:, 0]
        else:
            raise TypeError("expected Z or (X, Y, Z)")
        return x, y, z

    @staticmethod
    def _levels(z, levels):
        if np.ndim(levels) == 0:
            finite = z[np.isfinite(z)]
            ticks, _ = nice_ticks(finite.min(), finite.max(), int(levels) + 1)
            # ensure the levels cover the data range
            step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
            lo = ticks[0] - step if ticks[0] > finite.min() else ticks[0]
            hi = ticks[-1] + step if ticks[-1] < finite.max() else ticks[-1]
            return np.arange(lo, hi + step * 0.5, step)
        return np.asarray(levels, dtype=np.float64)

    def contour(self, *args, levels=8, colors=None, cmap=None, linestyles=None, linewidth=1,
                label=None):
        """Contour lines (marching squares). Single-color negative levels are dashed."""
        x, y, z = self._grid_args(args)
        lv = self._levels(z, levels)
        if np.ndim(levels) == 0:
            lv = lv[(lv > np.nanmin(z)) & (lv < np.nanmax(z))]
        if colors is not None:
            cols = self._colors(colors, len(lv), self.style.fg)
            cols = [cols[i % len(cols)] for i in range(len(lv))]
            single = len(set(cols)) == 1
        else:
            ramp = self.style.ramp(cmap)
            denom = (lv[-1] - lv[0]) or 1.0
            cols = [ramp.color_at((v - lv[0]) / denom) for v in lv]
            single = False
        if linestyles is None:
            styles = ["--" if (single and v < 0) else "-" for v in lv]
        elif isinstance(linestyles, str):
            styles = [linestyles] * len(lv)
        else:
            styles = [linestyles[i % len(linestyles)] for i in range(len(lv))]
        return self._add(A.ContourLines(x, y, z, lv, cols, styles, linewidth, label))

    def contourf(self, *args, levels=8, cmap=None, dither=None):
        """Filled contours; each band gets its own (dithered) color."""
        x, y, z = self._grid_args(args)
        lv = self._levels(z, levels)
        ramp = self.style.ramp(cmap)
        return self._add(A.FilledContour(x, y, z, lv, ramp,
                                         self.style.dither if dither is None else dither))

    def quiver(self, X, Y, U, V, C=None, color=None, cmap=None, scale=None, pivot="tail",
               label=None):
        """Vector field arrows. ``scale`` is pixels per data unit (auto by default)."""
        X, Y, U, V = (np.asarray(a, dtype=np.float64) for a in (X, Y, U, V))
        if X.ndim == 1 and Y.ndim == 1 and U.ndim == 2:
            X, Y = np.meshgrid(X, Y)
        cyc = self._cycle()
        if C is not None:
            ramp = self.style.ramp(cmap)
            c = np.asarray(C, dtype=np.float64).ravel()
            cols: Any = ramp.map(make_norm(c)(c)[None, :], dither=False)[0]
        else:
            cols = self._color(color, cyc["color"])
        return self._add(A.Quiver(X, Y, U, V, cols, scale, pivot, label))

    def pie(self, values, labels=None, colors=None, autopct=None, startangle=90.0,
            counterclock=True, depth=None, outline="auto"):
        """Pie chart. ``depth`` (pixels or ``"auto"``) tilts the pie and adds a 3-D extrusion."""
        st = self.style
        n = len(values)
        if colors is None:
            cols = [st.series_color(i) for i in range(n)]
        else:
            cols = [st.color(c) for c in colors]
        if depth is None:
            depth = "auto" if st.bar_shading else 0
        if outline == "auto":
            outline = st.bar_outline if st.bar_outline not in (None, "auto") else (
                "#000000" if st.bar_shading else None)
        self.axis("off")
        return self._add(A.Pie(values, labels, cols, autopct, startangle, counterclock, depth,
                               outline))

    def boxplot(self, data, labels=None, positions=None, widths=0.5, color=None,
                showfliers=True, label=None):
        """Box-and-whisker plot of one or several datasets."""
        datasets = _as_datasets(data)
        n = len(datasets)
        pos = np.arange(n, dtype=np.float64) if positions is None else np.asarray(positions)
        if color is None:
            cols = [self.style.series_color(i) for i in range(n)]
        else:
            cols = self._colors(color, n, self.style.series_color(0))
        if labels is not None:
            self.set_xticks(pos, labels)
        return self._add(A.BoxPlot(datasets, pos, widths, cols, showfliers, label))

    def violinplot(self, data, labels=None, positions=None, widths=0.7, color=None,
                   show_box=True, label=None):
        """Violin plot (Gaussian KDE evaluated per pixel row)."""
        datasets = _as_datasets(data)
        n = len(datasets)
        pos = np.arange(n, dtype=np.float64) if positions is None else np.asarray(positions)
        if color is None:
            cols = [self.style.series_color(i) for i in range(n)]
        else:
            cols = self._colors(color, n, self.style.series_color(0))
        if labels is not None:
            self.set_xticks(pos, labels)
        return self._add(A.Violins(datasets, pos, widths, cols, show_box, label))

    def text(self, x, y, s, color=None, ha="left", va="baseline", transform="data",
             fontsize="small", box=False):
        """Draw text. ``transform`` is ``"data"``, ``"axes"`` (0..1) or ``"pixels"``.

        ``fontsize`` is ``"small"`` (tick font) or ``"large"`` (label font).
        """
        st = self.style
        font = st.get_font() if fontsize == "large" else st.get_tick_font()
        return self._add(A.Text(x, y, s, self._color(color, st.fg), ha, va, transform, font,
                                box))

    def annotate(self, text, xy, xytext=None, color=None, arrow=True, ha="center",
                 va="bottom", fontsize="small"):
        """Label the point ``xy`` with ``text`` placed at ``xytext`` and an arrow."""
        st = self.style
        font = st.get_font() if fontsize == "large" else st.get_tick_font()
        if xytext is None:
            xytext, arrow = xy, False
        return self._add(A.Annotation(text, xy, xytext, self._color(color, st.fg), arrow,
                                      ha=ha, va=va, font=font))

    # ------------------------------------------------------------------ limits
    def _limits(self):
        xlog, ylog = self._xscale.log, self._yscale.log
        xb, yb, sx, sy = [], [], [], []
        for a in self.artists:
            b = a.bounds(xlog, ylog)
            if b is None:
                continue
            if b[0] is not None:
                xb += [b[0], b[1]]
            if b[2] is not None:
                yb += [b[2], b[3]]
            sx += list(a.sticky_x)
            sy += list(a.sticky_y)
        if self._xcats and not xb:
            xb = [0, len(self._xcats) - 1]
        if self._ycats and not yb:
            yb = [0, len(self._ycats) - 1]
        mx = 0.1 if self._xcats and not sx else self._margins[0]
        xlim = _resolve_lim(self._xlim, xb, sx, mx, self._xscale)
        ylim = _resolve_lim(self._ylim, yb, sy, self._margins[1], self._yscale)
        if self._invert[0]:
            xlim = xlim[::-1]
        if self._invert[1] and self._ylim == [None, None]:
            ylim = ylim[::-1]
        return xlim, ylim

    # ------------------------------------------------------------------ ticks
    def _tick_info(self, axis: str, lim, length: int, font):
        scale = self._xscale if axis == "x" else self._yscale
        user = self._xticks if axis == "x" else self._yticks
        user_labels = self._xticklabels if axis == "x" else self._yticklabels
        cats = self._xcats if axis == "x" else self._ycats
        lo, hi = sorted(lim)
        st = self.style
        if user is not None:
            vals = user
            labels = user_labels if user_labels is not None else format_ticks(vals, scale.log)
        elif cats:
            vals = np.array(list(cats.values()), dtype=np.float64)
            labels = list(cats.keys())
        else:
            vals = labels = None
        if vals is not None:
            tol = (hi - lo) * 1e-9
            keep = (vals >= lo - tol) & (vals <= hi + tol)
            labels = [st.text(lb) for lb, k in zip(labels, keep) if k]
            vals = vals[keep]
            return self._thin(axis, vals, labels, length, font)
        spacing = 5 * font.measure("0")[0] + 3 if axis == "x" else font.line_height * 2 + 4
        max_ticks = max(2, length // spacing + 1)
        while True:
            if scale.log:
                vals = log_ticks(lo, hi, max_ticks)
            else:
                vals, _ = nice_ticks(lo, hi, max_ticks)
            labels = [st.text(s) for s in format_ticks(vals, scale.log)]
            if axis == "y" or max_ticks <= 2:
                break
            widths = sum(font.measure(s)[0] for s in labels) + 4 * len(labels)
            if widths <= length:
                break
            max_ticks -= 1
        return vals, labels

    def _thin(self, axis, vals, labels, length, font):
        """Drop every other label until categorical/user labels no longer overlap."""
        if len(vals) < 2:
            return vals, labels
        stride = 1
        while stride < len(vals):
            sel = slice(None, None, stride)
            if axis == "x":
                need = sum(font.measure(s)[0] + 3 for s in labels[sel])
            else:
                need = (font.line_height + 1) * len(labels[sel])
            if need <= length:
                break
            stride += 1
        return vals, [lb if i % stride == 0 else "" for i, lb in enumerate(labels)]

    # ------------------------------------------------------------------ rendering
    def render(self, canvas: Canvas, cell: tuple[int, int, int, int]) -> None:
        st = self.style
        font = st.get_font()
        tfont = st.get_tick_font()
        shadow = st.color(st.text_shadow) if st.text_shadow is not None else None
        fg = st.fg
        if st.panel:
            draw_panel(canvas, st, cell, st.panel_fill, st.panel_border, st.panel_radius,
                       st.panel_shadow)
            inset = len(st.panel_border) + 2
        else:
            inset = 1
        cx0, cy0, cx1, cy1 = cell
        content = (cx0 + inset, cy0 + inset, cx1 - inset - (1 if st.panel_shadow else 0),
                   cy1 - inset - (1 if st.panel_shadow else 0))
        xlim, ylim = self._limits()
        title = st.text(self._title) if self._title else None
        title_h = font.measure(title)[1] + st.pad if title else 0

        if not self._axis_on:
            plot = (content[0], content[1] + title_h, content[2], content[3])
            xt = yt = np.array([])
            xtl = ytl = []
        else:
            xlabel = st.text(self._xlabel) if self._xlabel else None
            ylabel = st.text(self._ylabel) if self._ylabel else None
            xlabel_h = font.measure(xlabel)[1] + st.pad if xlabel else 0
            ylabel_w = font.measure(ylabel)[1] + st.pad if ylabel else 0
            tl = st.tick_length
            cap = tfont.ascent
            plot = (content[0] + ylabel_w + 24, content[1] + title_h + 4, content[2] - 8,
                    content[3] - xlabel_h - tl - tfont.line_height - 2)
            for _ in range(3):
                pw = max(plot[2] - plot[0], 8)
                ph = max(plot[3] - plot[1], 8)
                xt, xtl = self._tick_info("x", xlim, pw, tfont)
                yt, ytl = self._tick_info("y", ylim, ph, tfont)
                ytw = max((tfont.measure(s)[0] for s in ytl), default=0)
                xth = max((tfont.measure(s)[1] for s in xtl if s), default=0)
                last_w = tfont.measure(xtl[-1])[0] if xtl else 0
                cbar_w = self._colorbar_width(tfont, font, ph) if self._colorbar else 0
                left = content[0] + ylabel_w + (ytw + _TICK_GAP if ytw else 0) + tl + 1
                top = content[1] + max(title_h, cap // 2 + 1 if ytl else 0) + 1
                right = content[2] - max(last_w // 2 + 1, 1) - cbar_w
                if cbar_w:
                    right = content[2] - cbar_w
                bottom = content[3] - xlabel_h - (xth + 1 if xth else 0) - tl - 1
                plot = (left, top, max(right, left + 8), max(bottom, top + 8))
            plot = self._apply_aspect(plot, xlim, ylim)
            # final ticks for the final size
            xt, xtl = self._tick_info("x", xlim, plot[2] - plot[0], tfont)
            yt, ytl = self._tick_info("y", ylim, plot[3] - plot[1], tfont)

        self.last_rect = plot
        ctx = RenderContext(canvas, st, plot, xlim, ylim, self._xscale, self._yscale)
        px0, py0, px1, py1 = plot

        if st.plot_fill is not None:
            fill_area(canvas, st, plot, st.plot_fill)
        if self._grid and self._axis_on:
            gc = st.color(st.grid_color, st.foreground)
            with canvas.clip(*plot):
                if self._grid in ("both", "x"):
                    for v in xt:
                        x = float(ctx.tx(v))
                        canvas.polyline([x, x], [py0, py1 - 1], gc, dash=st.grid_linestyle)
                if self._grid in ("both", "y"):
                    for v in yt:
                        y = float(ctx.ty(v))
                        canvas.polyline([px0, px1 - 1], [y, y], gc, dash=st.grid_linestyle)

        with canvas.clip(*plot):
            for art in sorted(self.artists, key=lambda a: a.zorder):
                art.draw(ctx)

        if self._axis_on:
            fc = st.color(st.frame_color, st.foreground)
            canvas.rect(px0 - 1, py0 - 1, px1 + 1, py1 + 1, fc)
            tl = st.tick_length
            boxes = []
            for v, s in zip(xt, xtl):
                x = int(round(float(ctx.tx(v))))
                canvas.fill_rect(x, py1 + 1, x + 1, py1 + 1 + tl, fc)
                if s:
                    w = tfont.measure(s)[0]
                    lx = min(max(x - w // 2, content[0]), content[2] - w)
                    boxes.append(canvas.text(lx, py1 + 2 + tl, s, tfont, fg, "left", "top",
                                             shadow=shadow))
            cap = tfont.ascent
            for v, s in zip(yt, ytl):
                y = int(round(float(ctx.ty(v))))
                canvas.fill_rect(px0 - 1 - tl, y, px0 - 1, y + 1, fc)
                if not s:
                    continue
                right = px0 - 1 - tl - _TICK_GAP  # exclusive right edge of the label
                box = (right - tfont.measure(s)[0], y - cap // 2, right, y - cap // 2 + cap)
                if any(_overlaps(box, b) for b in boxes):
                    continue  # e.g. the lowest y label colliding with the first x label
                canvas.text(right - 1, y, s, tfont, fg, "right", "center", shadow=shadow)
            if self._xlabel:
                canvas.text((px0 + px1) // 2, content[3] - font.measure(st.text(self._xlabel))[1],
                            st.text(self._xlabel), font, fg, "center", "top", shadow=shadow)
            if self._ylabel:
                canvas.text(content[0], (py0 + py1) // 2, st.text(self._ylabel), font, fg,
                            "left", "center", rotation=90, shadow=shadow)
        if title:
            tc = st.color(st.title_color, st.foreground)
            w = font.measure(title)[0]
            tx = min(max((px0 + px1) // 2 - w // 2, content[0]), max(content[2] - w, content[0]))
            canvas.text(tx, content[1], title, font, tc, "left", "top", shadow=shadow)
        if self._colorbar is not None:
            self._draw_colorbar(canvas, plot, tfont, font, shadow)
        if self._legend is not None:
            self._draw_legend(canvas, ctx, tfont, shadow)

    def _apply_aspect(self, plot, xlim, ylim):
        if self._aspect == "auto":
            return plot
        ratio = 1.0 if self._aspect == "equal" else float(self._aspect)
        x0, y0, x1, y1 = plot
        w, h = x1 - x0, y1 - y0
        fx = self._xscale.fwd(np.array(xlim))
        fy = self._yscale.fwd(np.array(ylim))
        dx, dy = abs(fx[1] - fx[0]), abs(fy[1] - fy[0]) * ratio
        if dx == 0 or dy == 0:
            return plot
        if dx / w > dy / h:
            nh = max(int(round(dy * w / dx)), 4)
            off = (h - nh) // 2
            return (x0, y0 + off, x1, y0 + off + nh)
        nw = max(int(round(dx * h / dy)), 4)
        off = (w - nw) // 2
        return (x0 + off, y0, x0 + off + nw, y1)

    # ------------------------------------------------------------------ colorbar
    _CBAR_GAP = 4
    _CBAR_W = 6

    def _colorbar_width(self, tfont, font, height):
        cb = self._colorbar
        st = self.style
        _, labels = cb.tick_values(max(2, height // (tfont.line_height * 2 + 4) + 1))
        lw = max((tfont.measure(s)[0] for s in labels), default=0)
        label_w = font.measure(st.text(cb.label))[1] + st.pad if cb.label else 0
        return (self._CBAR_GAP + self._CBAR_W + 2 + st.tick_length + _TICK_GAP + lw + label_w
                + 1)

    def _draw_colorbar(self, canvas, plot, tfont, font, shadow):
        st = self.style
        cb = self._colorbar
        m = cb.mappable
        px0, py0, px1, py1 = plot
        bx0 = px1 + 1 + self._CBAR_GAP
        bx1 = bx0 + self._CBAR_W
        h = py1 - py0
        rows = np.arange(h)
        t = 1.0 - rows / max(h - 1, 1)
        levels = getattr(m, "levels", None)
        if levels is not None and len(levels) > 1:
            n = len(levels) - 1
            band = np.clip(np.floor(t * n), 0, n - 1)
            t = (band + 0.5) / n
        tt = np.repeat(t[:, None], self._CBAR_W, axis=1)
        img = m.ramp.map(tt, getattr(m, "dither", True), bx0, py0)
        canvas.blit(img, bx0, py0)
        fc = st.color(st.frame_color, st.foreground)
        canvas.rect(bx0 - 1, py0 - 1, bx1 + 1, py1 + 1, fc)
        vals, labels = cb.tick_values(max(2, h // (tfont.line_height * 2 + 4) + 1))
        tl = st.tick_length
        lw = 0
        for v, s in zip(vals, labels):
            if levels is not None and len(levels) > 1:
                u = float(np.interp(v, levels, np.linspace(0, 1, len(levels))))
            else:
                u = float(m.norm(np.array([v]))[0])
            y = int(round(py0 + (1 - u) * (h - 1)))
            canvas.fill_rect(bx1 + 1, y, bx1 + 1 + tl, y + 1, fc)
            canvas.text(bx1 + 1 + tl + _TICK_GAP, y, st.text(s), tfont, st.fg, "left", "center",
                        shadow=shadow)
            lw = max(lw, tfont.measure(st.text(s))[0])
        if cb.label:
            canvas.text(bx1 + 1 + tl + _TICK_GAP + lw + st.pad, (py0 + py1) // 2,
                        st.text(cb.label), font,
                        st.fg, "left", "center", rotation=90, shadow=shadow)

    # ------------------------------------------------------------------ legend
    def _draw_legend(self, canvas, ctx, font, shadow):
        st = self.style
        entries = [e for a in self.artists for e in a.legend_entries()]
        if not entries:
            return
        title = st.text(self._legend.get("title")) if self._legend.get("title") else None
        handle_w = 9
        lh = font.line_height
        labels = [st.text(e["label"]) for e in entries]
        text_w = max(font.measure(s)[0] for s in labels)
        if title:
            text_w = max(text_w, font.measure(title)[0] - handle_w - 2)
        w = 3 + handle_w + 2 + text_w + 3
        h = 3 + (lh + 1) * (len(entries) + (1 if title else 0)) + 1
        px0, py0, px1, py1 = ctx.rect
        loc = self._legend["loc"]
        if loc == "best":
            loc = self._best_loc(ctx, w, h)
        fx, fy = _LOCS[loc]
        lx = int(px0 + 2 + fx * (px1 - px0 - 4 - w))
        ly = int(py0 + 2 + fy * (py1 - py0 - 4 - h))
        fill = st.legend_fill or st.panel_fill or st.background
        draw_panel(canvas, st, (lx, ly, lx + w, ly + h), fill, (st.foreground,),
                   1 if st.panel_radius else 0, None)
        y = ly + 3
        if title:
            canvas.text(lx + 3, y, title, font, st.color(st.title_color, st.foreground),
                        shadow=shadow)
            y += lh + 1
        for e, s in zip(entries, labels):
            A.legend_handle(canvas, st, e, lx + 3, y, handle_w, font.ascent)
            canvas.text(lx + 3 + handle_w + 2, y, s, font, st.fg, shadow=shadow)
            y += lh + 1

    def _best_loc(self, ctx, w, h):
        xs, ys = [], []
        for a in self.artists:
            p = a.points()
            if p is not None:
                xs.append(np.asarray(ctx.tx(p[0]), dtype=np.float64).ravel())
                ys.append(np.asarray(ctx.ty(p[1]), dtype=np.float64).ravel())
        if not xs:
            return "upper right"
        px, py = np.concatenate(xs), np.concatenate(ys)
        # densify line segments a bit so long lines through a corner count
        px0, py0, px1, py1 = ctx.rect
        best, best_score = "upper right", None
        for loc in ("upper right", "upper left", "lower right", "lower left"):
            fx, fy = _LOCS[loc]
            lx = px0 + 2 + fx * (px1 - px0 - 4 - w)
            ly = py0 + 2 + fy * (py1 - py0 - 4 - h)
            inside = (px >= lx - 2) & (px <= lx + w + 2) & (py >= ly - 2) & (py <= ly + h + 2)
            score = int(inside.sum())
            if best_score is None or score < best_score:
                best, best_score = loc, score
        return best


def _resolve_lim(user, bounds, sticky, margin, scale: Scale):
    if user[0] is not None and user[1] is not None:
        return (float(user[0]), float(user[1]))
    vals = np.asarray(bounds, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if scale.log:
        vals = vals[vals > 0]
    if vals.size == 0:
        lo, hi = (1.0, 10.0) if scale.log else (0.0, 1.0)
    else:
        lo, hi = float(vals.min()), float(vals.max())
    flo, fhi = float(scale.fwd(lo)), float(scale.fwd(hi))
    if flo == fhi:
        d = 0.5 if scale.log else (abs(flo) * 0.1 or 0.5)
        flo, fhi = flo - d, fhi + d
    else:
        span = fhi - flo
        nlo, nhi = flo - margin * span, fhi + margin * span
        fs = scale.fwd(np.asarray(list(sticky), dtype=np.float64))
        fs = fs[np.isfinite(fs)]
        tol = span * 1e-9
        below = fs[(fs > nlo) & (fs <= flo + tol)]
        above = fs[(fs < nhi) & (fs >= fhi - tol)]
        flo = float(below.max()) if below.size else nlo
        fhi = float(above.min()) if above.size else nhi
    lo, hi = float(scale.inv(flo)), float(scale.inv(fhi))
    if user[0] is not None:
        lo = float(user[0])
    if user[1] is not None:
        hi = float(user[1])
    return (lo, hi)


def _centers_to_edges(c):
    c = np.asarray(c, dtype=np.float64)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _as_datasets(data):
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return [data]
        return [data[:, i] for i in range(data.shape[1])]
    if len(data) and np.ndim(data[0]) == 0:
        return [np.asarray(data, dtype=np.float64)]
    return [np.asarray(d, dtype=np.float64) for d in data]


# ---------------------------------------------------------------------- panels
def is_gradient(fill) -> bool:
    return (isinstance(fill, (tuple, list)) and len(fill) == 2
            and all(isinstance(f, (str, tuple, list)) for f in fill))


def fill_area(canvas: Canvas, style, rect, fill, mask: np.ndarray | None = None):
    """Fill ``rect`` with a solid color or a dithered ``(top, bottom)`` gradient."""
    if fill is None:
        return
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    if is_gradient(fill):
        ramp = _gradient_ramp(style, fill)
        t = np.repeat((np.arange(h) / max(h - 1, 1))[:, None], w, axis=1)
        img = ramp.map(t, style.dither, x0, y0)
        canvas.blit(img, x0, y0, mask=mask)
    else:
        c = style.color(fill)
        if mask is None:
            canvas.fill_rect(x0, y0, x1, y1, c)
        else:
            canvas.fill_mask(mask, x0, y0, c)


def _gradient_ramp(style, fill) -> Ramp:
    from .colormaps import Colormap, build_ramp

    cm = Colormap("gradient", [style.color(fill[0]), style.color(fill[1])])
    pal = style.colors.palette
    if pal is not None and len(pal) <= 4:
        return Ramp(np.array([0.0, 1.0]), np.array([style.color(fill[0]),
                                                    style.color(fill[1])], dtype=np.uint8))
    return build_ramp(cm, style.colors, style.gradient_levels)


def rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    m = np.ones((h, w), dtype=bool)
    r = min(r, w // 2, h // 2)
    if r <= 0:
        return m
    yy, xx = np.mgrid[0:r, 0:r]
    dx = r - (xx + 0.5)
    dy = r - (yy + 0.5)
    corner = dx * dx + dy * dy <= (r - 0.5) ** 2
    m[:r, :r] &= corner
    m[:r, w - r :] &= corner[:, ::-1]
    m[h - r :, :r] &= corner[::-1, :]
    m[h - r :, w - r :] &= corner[::-1, ::-1]
    return m


def draw_panel(canvas: Canvas, style, rect, fill, border, radius=0, shadow=None):
    """Retro UI window: optional drop shadow, concentric border rings and (gradient) fill."""
    x0, y0, x1, y1 = rect
    if shadow is not None:
        x1 -= 1
        y1 -= 1
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return
    if shadow is not None:
        canvas.fill_mask(rounded_mask(w, h, radius), x0 + 1, y0 + 1, style.color(shadow))
    n = len(border)
    for i, c in enumerate(border):
        outer = rounded_mask(w - 2 * i, h - 2 * i, max(radius - i, 0))
        canvas.fill_mask(outer, x0 + i, y0 + i, style.color(c))
    inner = rounded_mask(w - 2 * n, h - 2 * n, max(radius - n, 0))
    fill_area(canvas, style, (x0 + n, y0 + n, x1 - n, y1 - n), fill, mask=inner)
