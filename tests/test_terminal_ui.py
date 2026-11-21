import asyncio
import io

import numpy as np
import pytest

import pixelplotlib as ppl

pytest.importorskip("rich")
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from pixelplotlib.rich import Plot  # noqa: E402


def decode(lines) -> np.ndarray:
    """Turn half-block segment lines back into an RGB array (default colours -> -1)."""
    rows = []
    for line in lines:
        top, bottom = [], []
        for seg in line:
            fg = seg.style.color.get_truecolor() if seg.style.color else (-1, -1, -1)
            bg = seg.style.bgcolor.get_truecolor() if seg.style.bgcolor else (-1, -1, -1)
            for ch in seg.text:
                assert ch in " \u2580"
                top.append(bg if ch == " " else fg)
                bottom.append(bg)
        rows += [top, bottom]
    return np.array(rows, dtype=int)


def render(renderable, width=80, height=None):
    console = Console(width=width, color_system="truecolor", file=io.StringIO())
    options = console.options.update(height=height) if height else console.options
    return console.render_lines(renderable, options, pad=False)


@pytest.fixture
def fig():
    fig, ax = ppl.subplots(size=(60, 25), style="pico8")
    ax.plot([0, 1, 2], [1, 0, 1], marker="heart")
    ax.set(title="HI")
    return fig


def test_figure_prints_pixel_exact_with_odd_height(fig):
    img = decode(render(fig))
    np.testing.assert_array_equal(img[:25], fig.to_array())
    assert (img[25] == -1).all()  # odd last row: bottom half shows the terminal background


@pytest.mark.parametrize(
    "kwargs, width, height, expected",
    [
        ({}, 40, None, (40, 17)),  # shrunk to the console, keeping the aspect ratio
        ({}, 80, 5, (24, 10)),  # shrunk to a fixed container height
        ({"width": 30}, 80, None, (30, 12)),
        ({"height": 4}, 80, None, (19, 8)),
        ({"width": 30, "height": 3}, 80, None, (30, 6)),
        ({"expand": True}, 70, None, (70, 29)),
        ({"expand": True}, 70, 6, (70, 12)),
    ],
)
def test_plot_size(fig, kwargs, width, height, expected):
    img = decode(render(Plot(fig, **kwargs), width, height))
    w, h = expected
    assert img.shape[:2] == (h + h % 2, w)
    np.testing.assert_array_equal(img[:h], fig.to_array(size=(w, h)))
    assert fig.size == (60, 25)


def test_figure_shrinks_inside_table(fig):
    table = Table(show_header=False)
    table.add_row(fig, Plot(fig, width=20))
    lines = render(table, width=50)
    assert {sum(seg.cell_length for seg in line) for line in lines} == {50}


def test_textual_widget_fills_and_follows_its_size(fig):
    pytest.importorskip("textual")
    from textual.app import App

    from pixelplotlib.textual import PixelPlot

    class PlotApp(App):
        CSS = "#auto { width: auto; height: auto; }"

        def compose(self):
            yield PixelPlot(fig)
            yield PixelPlot(fig, id="auto")

    def shown(widget):
        return decode([list(widget.render_line(y)) for y in range(widget.size.height)])

    async def run():
        async with PlotApp().run_test(size=(70, 30)) as pilot:
            plot, auto = pilot.app.query(PixelPlot)
            await pilot.pause()
            assert tuple(plot.size) == (70, 17)
            np.testing.assert_array_equal(shown(plot), fig.to_array(size=(70, 34)))
            assert tuple(auto.size) == (60, 13)  # native width, height from aspect ratio

            await pilot.resize_terminal(50, 30)
            await pilot.pause()
            np.testing.assert_array_equal(shown(plot), fig.to_array(size=(50, 34)))

            new, ax = ppl.subplots(style="gameboy")
            ax.bar([1, 2, 3], [3, 1, 2])
            plot.figure = new
            await pilot.pause()
            np.testing.assert_array_equal(shown(plot), new.to_array(size=(50, 34)))

    asyncio.run(run())
