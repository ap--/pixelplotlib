"""Draw figures with `Rich <https://github.com/Textualize/rich>`_ in any colour terminal.

Each terminal cell shows two vertically stacked pixels using the upper half block
character ``▀`` (foreground = top pixel, background = bottom pixel), so this works
wherever Rich does, with no sixel support needed. Figures are re-rendered at the size
they are drawn at, so fonts and one-pixel lines stay sharp at any terminal width::

    from rich import print
    print(fig)                                  # a Figure is a Rich renderable
    print(Panel(Plot(fig, width=60)))           # or wrap it to choose the size

Requires the ``rich`` extra: ``pip install "pixelplotlib[rich]"``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.color import Color
from rich.console import Console, ConsoleOptions
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style

from .figure import Figure

__all__ = ["Plot", "halfblock_lines"]

#: Narrowest width (in cells) a :class:`Plot` without an explicit width shrinks to.
MIN_WIDTH = 16

_UPPER_HALF = "\u2580"


class Plot:
    """A Rich renderable that draws a :class:`~pixelplotlib.Figure` with half blocks.

    Parameters
    ----------
    figure:
        The figure to draw. It is rendered whenever the renderable is printed, so later
        changes to the figure show up the next time it is drawn.
    width, height:
        Size in terminal cells (one cell is one pixel wide and two pixels tall). If only
        one is given the other follows the figure's aspect ratio. By default the figure
        is drawn at its native resolution, shrunk to fit the available width (and height,
        when the container has a fixed height such as a :class:`rich.layout.Layout`).
    expand:
        Fill the available width (and height, if the container fixes it) instead.
    """

    def __init__(self, figure: Figure, width: int | None = None, height: int | None = None,
                 *, expand: bool = False):
        if not isinstance(figure, Figure):
            raise TypeError(f"expected a pixelplotlib Figure, got {type(figure).__name__}")
        self.figure = figure
        self.width = width
        self.height = height
        self.expand = expand

    def pixel_size(self, max_width: int, max_height: int | None = None) -> tuple[int, int]:
        """``(width, height)`` in pixels to render at, given the space available in cells."""
        fw, fh = self.figure.size
        max_width = max(max_width, 1)
        w, h = self.width, self.height
        if w is None and h is None:
            w = max_width if self.expand else fw
            h = max_height if self.expand else None
        if w is None:
            w = round(h * 2 * fw / fh)
        w = max(min(w, max_width), 1)
        px_h = max(h * 2 if h is not None else round(w * fh / fw), 1)
        if self.height is None and max_height is not None and px_h > 2 * max_height:
            px_h = 2 * max(max_height, 1)
            if self.width is None and not self.expand:
                w = max(min(w, round(px_h * fw / fh)), 1)
        return w, px_h

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Iterator[Segment]:
        size = self.pixel_size(options.max_width, options.height)
        pixels = self.figure.to_array(size=size)
        for line in halfblock_lines(pixels):
            yield from line
            yield Segment.line()

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        max_width = options.max_width
        if self.width is not None:
            width = min(self.width, max_width)
            return Measurement(width, width)
        if self.expand:
            return Measurement(max_width, max_width)
        natural = self.pixel_size(max_width)[0]
        return Measurement(min(MIN_WIDTH, natural), natural)

    def __repr__(self) -> str:
        return f"<Plot {self.figure!r} width={self.width} height={self.height}>"


def halfblock_lines(pixels: np.ndarray) -> list[list[Segment]]:
    """Convert an ``(H, W, 3)`` uint8 image into ``ceil(H / 2)`` lines of Rich segments.

    Runs of identical cells share one segment. An odd last pixel row is drawn over the
    terminal's default background.
    """
    pixels = np.asarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {pixels.shape}")
    packed = (pixels[..., 0].astype(np.int64) << 16) | (pixels[..., 1].astype(np.int64) << 8)
    packed |= pixels[..., 2]
    if packed.shape[0] % 2:
        packed = np.vstack([packed, np.full((1, packed.shape[1]), -1, dtype=np.int64)])
    top, bottom = packed[0::2], packed[1::2]
    styles: dict[tuple[int, int], tuple[str, Style]] = {}
    lines = []
    for t, b in zip(top, bottom):
        starts = np.flatnonzero((np.diff(t, prepend=-2) != 0) | (np.diff(b, prepend=-2) != 0))
        lengths = np.diff(np.append(starts, len(t)))
        line = []
        for key, n in zip(zip(t[starts].tolist(), b[starts].tolist()), lengths.tolist()):
            if key not in styles:
                styles[key] = _cell_style(*key)
            char, style = styles[key]
            line.append(Segment(char * n, style))
        lines.append(line)
    return lines


def _color(packed: int) -> Color:
    return Color.from_rgb(packed >> 16, (packed >> 8) & 255, packed & 255)


def _cell_style(top: int, bottom: int) -> tuple[str, Style]:
    if bottom < 0:
        return _UPPER_HALF, Style(color=_color(top))
    if top == bottom:
        return " ", Style(bgcolor=_color(top))
    return _UPPER_HALF, Style(color=_color(top), bgcolor=_color(bottom))
