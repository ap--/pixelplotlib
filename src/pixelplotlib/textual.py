"""A `Textual <https://github.com/Textualize/textual>`_ widget that shows a figure.

The widget re-renders its figure at exactly the widget's size (one pixel per cell
column, two per cell row), so plots stay crisp as the terminal is resized::

    from textual.app import App
    from pixelplotlib.textual import PixelPlot

    class PlotApp(App):
        def compose(self):
            fig, ax = ppl.subplots(style="pico8")
            ax.plot([1, 3, 2, 4])
            yield PixelPlot(fig)

Assign a new figure to :attr:`PixelPlot.figure` to replace the plot, or call
:meth:`~textual.widget.Widget.refresh` after changing the current figure in place.

Requires the ``textual`` extra: ``pip install "pixelplotlib[textual]"``.
"""

from __future__ import annotations

from rich.console import RenderableType
from textual.geometry import Size
from textual.reactive import reactive
from textual.widget import Widget

from .figure import Figure
from .rich import Plot
from .styles import Style

__all__ = ["PixelPlot"]


class PixelPlot(Widget):
    """Display a :class:`~pixelplotlib.Figure`, filling the widget.

    Parameters
    ----------
    figure:
        The figure to show. Defaults to a new empty figure in ``style``; draw into it
        through ``widget.figure`` and call ``widget.refresh()``.
    style:
        Style for the default figure (ignored when ``figure`` is given).

    With ``width: auto`` / ``height: auto`` in CSS the widget takes the figure's native
    width (one cell per pixel) and the matching height for its aspect ratio.
    """

    DEFAULT_CSS = """
    PixelPlot {
        width: 1fr;
        height: 1fr;
    }
    """

    figure: reactive[Figure] = reactive(Figure, layout=True, always_update=True)
    """The figure being displayed; assigning a figure (even the same one) redraws it."""

    def __init__(self, figure: Figure | None = None, *, style: str | Style | None = None,
                 name: str | None = None, id: str | None = None, classes: str | None = None,
                 disabled: bool = False):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.set_reactive(PixelPlot.figure, figure if figure is not None else Figure(style=style))

    def render(self) -> RenderableType:
        width, height = self.content_size
        if width <= 0 or height <= 0:
            return ""
        return Plot(self.figure, width=width, height=height)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return self.figure.width

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        _, px_h = Plot(self.figure, width=width).pixel_size(max(width, 1))
        return (px_h + 1) // 2
