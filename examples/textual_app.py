"""A live-updating pixelplotlib dashboard in the terminal, built with Textual.

Run with ``uv run --extra textual python examples/textual_app.py``. Press ``s`` to cycle
through the styles and ``q`` to quit; the plots re-render as you resize the terminal.
"""

from __future__ import annotations

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header

import pixelplotlib as ppl
from pixelplotlib.textual import PixelPlot


class Dashboard(App):
    TITLE = "pixelplotlib"
    BINDINGS = [("s", "next_style", "Next style"), ("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.style_names = ppl.available()
        self.style_index = 0
        self.t = 0.0
        self.rng = np.random.default_rng(0)

    @property
    def plot_style(self) -> str:
        return self.style_names[self.style_index]

    def compose(self) -> ComposeResult:
        yield Header()
        yield PixelPlot(id="waves")
        with Horizontal():
            yield PixelPlot(id="hist")
            yield PixelPlot(id="heatmap")
        yield Footer()

    def on_mount(self) -> None:
        self.redraw()
        self.set_interval(1 / 10, self.tick)

    def tick(self) -> None:
        self.t += 0.1
        self.redraw()

    def action_next_style(self) -> None:
        self.style_index = (self.style_index + 1) % len(self.style_names)
        self.sub_title = self.plot_style
        self.redraw()

    def redraw(self) -> None:
        x = np.linspace(0, 4 * np.pi, 200)
        fig, ax = ppl.subplots(style=self.plot_style)
        ax.plot(x, np.sin(x - self.t), label="sin")
        ax.plot(x, np.cos(x + self.t / 2), label="cos")
        ax.set(title=f"LIVE WAVES ({self.plot_style})", xlabel="x")
        ax.legend(loc="lower left")
        self.query_one("#waves", PixelPlot).figure = fig

        fig, ax = ppl.subplots(style=self.plot_style)
        ax.hist(self.rng.normal(np.sin(self.t), 1, 400), bins=12)
        ax.set(title="HISTOGRAM")
        self.query_one("#hist", PixelPlot).figure = fig

        y, x = np.mgrid[-2:2:24j, -2:2:24j]
        fig, ax = ppl.subplots(style=self.plot_style)
        ax.imshow(np.sin(3 * x + self.t) * np.cos(3 * y), cmap="magma")
        ax.set(title="FIELD")
        self.query_one("#heatmap", PixelPlot).figure = fig


if __name__ == "__main__":
    Dashboard().run()
