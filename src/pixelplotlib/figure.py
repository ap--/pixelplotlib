"""Figures: canvas size, subplot layout and image export."""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np

from .axes import Axes, fill_area
from .canvas import Canvas
from .styles import Style, get_style


class Figure:
    """A pixel canvas holding one or more :class:`Axes`.

    Parameters
    ----------
    size:
        ``(width, height)`` in *logical* pixels. Defaults to the style's native resolution.
    style:
        Style name or :class:`Style` (see :func:`pixelplotlib.styles.available`).
    scale:
        Integer upscaling factor used when saving/showing.
    """

    def __init__(self, size: tuple[int, int] | None = None, style: str | Style | None = None,
                 scale: int | None = None):
        self.style = get_style(style)
        w, h = size or self.style.size
        self.size = (int(w), int(h))
        self.scale = int(scale or self.style.scale)
        self.axes: list[Axes] = []
        self._suptitle: str | None = None
        self._gca: Axes | None = None

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    # ------------------------------------------------------------------ layout
    def add_subplot(self, nrows: int = 1, ncols: int = 1, index: int | tuple[int, int] = 1) -> Axes:
        """Add axes in a grid cell (1-based ``index``; a ``(first, last)`` tuple spans cells)."""
        first, last = (index, index) if isinstance(index, int) else index
        if not (1 <= first <= nrows * ncols and 1 <= last <= nrows * ncols):
            raise ValueError("subplot index out of range")
        r0, c0 = divmod(first - 1, ncols)
        r1, c1 = divmod(last - 1, ncols)
        ax = Axes(self, ("grid", nrows, ncols, min(r0, r1), max(r0, r1), min(c0, c1),
                         max(c0, c1)))
        self.axes.append(ax)
        self._gca = ax
        return ax

    def add_axes(self, rect: tuple[float, float, float, float]) -> Axes:
        """Add axes at ``(left, bottom, width, height)`` in figure fractions."""
        ax = Axes(self, ("rect", *rect))
        self.axes.append(ax)
        self._gca = ax
        return ax

    def subplots(self, nrows: int = 1, ncols: int = 1, squeeze: bool = True):
        """Create a grid of axes; returns one Axes or a numpy array of them."""
        arr = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                arr[r, c] = self.add_subplot(nrows, ncols, r * ncols + c + 1)
        if squeeze:
            if arr.size == 1:
                return arr[0, 0]
            if nrows == 1 or ncols == 1:
                return arr.ravel()
        return arr

    def gca(self) -> Axes:
        if self._gca is None:
            return self.add_subplot()
        return self._gca

    def sca(self, ax: Axes) -> None:
        self._gca = ax

    def suptitle(self, title: str | None) -> Figure:
        self._suptitle = title
        return self

    def colorbar(self, mappable=None, ax: Axes | None = None, label: str | None = None,
                 ticks=None):
        """Add a colorbar next to ``ax`` (defaults to the axes owning ``mappable``)."""
        if ax is None:
            ax = next((a for a in self.axes if mappable in a.artists), None) or self.gca()
        return ax.colorbar(mappable, label=label, ticks=ticks)

    def _cell(self, ax: Axes, area) -> tuple[int, int, int, int]:
        gx0, gy0, gx1, gy1 = area
        spec = ax._spec
        if spec[0] == "rect":
            _, l, b, w, h = spec
            x0 = int(round(l * self.width))
            x1 = int(round((l + w) * self.width))
            y1 = int(round((1 - b) * self.height))
            y0 = int(round((1 - b - h) * self.height))
            return x0, y0, x1, y1
        _, nrows, ncols, r0, r1, c0, c1 = spec
        gap = 2 if self.style.panel else 4
        cw = (gx1 - gx0 - (ncols - 1) * gap) / ncols
        ch = (gy1 - gy0 - (nrows - 1) * gap) / nrows
        x0 = gx0 + c0 * (cw + gap)
        x1 = gx0 + (c1 + 1) * cw + c1 * gap
        y0 = gy0 + r0 * (ch + gap)
        y1 = gy0 + (r1 + 1) * ch + r1 * gap
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

    # ------------------------------------------------------------------ rendering
    def render(self) -> Canvas:
        """Rasterize the figure at its logical resolution."""
        st = self.style
        canvas = Canvas(self.width, self.height)
        fill_area(canvas, st, (0, 0, self.width, self.height), st.background)
        margin = 2
        top = margin
        if self._suptitle:
            font = st.get_font()
            text = st.text(self._suptitle)
            shadow = st.color(st.text_shadow) if st.text_shadow is not None else None
            canvas.text(self.width // 2, top + 1, text, font,
                        st.color(st.title_color, st.foreground), "center", "top", shadow=shadow)
            top += font.measure(text)[1] + 3
        area = (margin, top, self.width - margin, self.height - margin)
        for ax in self.axes:
            ax.render(canvas, self._cell(ax, area))
        return canvas

    def to_array(self, scale: int | None = None) -> np.ndarray:
        """Return the rendered figure as an ``(H, W, 3)`` uint8 array."""
        return self.render().to_array(scale or 1)

    def to_image(self, scale: int | None = None, scanlines: bool = False):
        """Return a :class:`PIL.Image.Image` upscaled with nearest-neighbour."""
        from PIL import Image

        s = scale or self.scale
        arr = self.render().to_array(s)
        if scanlines and s >= 2:
            arr = apply_scanlines(arr, s)
        return Image.fromarray(arr, "RGB")

    def save(self, path: str | os.PathLike | Any, scale: int | None = None,
             scanlines: bool = False, **kwargs) -> None:
        """Save to an image file (format from the extension; PNG recommended).

        ``scanlines=True`` darkens every ``scale``-th row for a CRT look.
        """
        img = self.to_image(scale, scanlines)
        img.save(path, **kwargs)

    savefig = save

    def _repr_png_(self) -> bytes:
        buf = io.BytesIO()
        self.to_image().save(buf, format="PNG")
        return buf.getvalue()

    def show(self, scale: int | None = None) -> None:
        """Display inline in Jupyter, otherwise open the system image viewer."""
        try:
            from IPython import get_ipython
            from IPython.display import Image as IPImage
            from IPython.display import display

            ip = get_ipython()
            if ip is not None and getattr(ip, "kernel", None) is not None:
                buf = io.BytesIO()
                self.to_image(scale).save(buf, format="PNG")
                display(IPImage(data=buf.getvalue()))
                return
        except ImportError:
            pass
        self.to_image(scale).show()

    def __repr__(self) -> str:
        return (f"<Figure {self.width}x{self.height} style={self.style.name!r} "
                f"axes={len(self.axes)}>")


def apply_scanlines(arr: np.ndarray, scale: int, strength: float = 0.35) -> np.ndarray:
    """Darken the last row of every ``scale``-tall pixel row (CRT effect)."""
    out = arr.copy()
    rows = np.arange(out.shape[0]) % scale == scale - 1
    out[rows] = (out[rows].astype(np.float32) * (1 - strength)).astype(np.uint8)
    return out
