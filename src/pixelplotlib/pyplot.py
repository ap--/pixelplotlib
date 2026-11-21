"""A small matplotlib.pyplot-like stateful interface."""

from __future__ import annotations

import functools

from .axes import Axes
from .figure import Figure
from .styles import Style

_figures: list[Figure] = []


def figure(size: tuple[int, int] | None = None, style: str | Style | None = None,
           scale: int | None = None) -> Figure:
    """Create a new figure and make it current."""
    fig = Figure(size=size, style=style, scale=scale)
    _figures.append(fig)
    return fig


def gcf() -> Figure:
    """Current figure (created on demand)."""
    if not _figures:
        return figure()
    return _figures[-1]


def gca() -> Axes:
    """Current axes (created on demand)."""
    return gcf().gca()


def subplots(nrows: int = 1, ncols: int = 1, size: tuple[int, int] | None = None,
             style: str | Style | None = None, scale: int | None = None, squeeze: bool = True):
    """Create a figure with a grid of axes: ``fig, ax = subplots()``."""
    fig = figure(size=size, style=style, scale=scale)
    return fig, fig.subplots(nrows, ncols, squeeze=squeeze)


def subplot(nrows: int, ncols: int, index) -> Axes:
    return gcf().add_subplot(nrows, ncols, index)


def close(fig: Figure | str | None = None) -> None:
    """Close the current figure, a given figure, or ``"all"``."""
    if fig == "all":
        _figures.clear()
    elif fig is None:
        if _figures:
            _figures.pop()
    elif fig in _figures:
        _figures.remove(fig)


def savefig(path, **kwargs) -> None:
    gcf().save(path, **kwargs)


def show(**kwargs) -> None:
    """Show and close all open figures (``backend="sixel"`` draws them in the terminal)."""
    for fig in list(_figures):
        fig.show(**kwargs)
    _figures.clear()


def suptitle(title: str) -> Figure:
    return gcf().suptitle(title)


def colorbar(mappable=None, label: str | None = None, ticks=None):
    return gca().colorbar(mappable, label=label, ticks=ticks)


def _wrap(name: str, target: str | None = None):
    meth = getattr(Axes, target or name)

    @functools.wraps(meth)
    def wrapper(*args, **kwargs):
        return getattr(gca(), target or name)(*args, **kwargs)

    wrapper.__name__ = name
    return wrapper


_AXES_FUNCS = [
    "plot", "step", "scatter", "bar", "barh", "hist", "hist2d", "fill_between", "fill",
    "errorbar", "stem", "axhline", "axvline", "axhspan", "axvspan", "imshow", "pcolormesh",
    "contour", "contourf", "quiver", "pie", "boxplot", "violinplot", "text", "annotate",
    "legend", "grid", "axis",
]
_RENAMED = {
    "title": "set_title", "xlabel": "set_xlabel", "ylabel": "set_ylabel",
    "xlim": "set_xlim", "ylim": "set_ylim", "xscale": "set_xscale", "yscale": "set_yscale",
    "xticks": "set_xticks", "yticks": "set_yticks",
}

for _name in _AXES_FUNCS:
    globals()[_name] = _wrap(_name)
for _name, _target in _RENAMED.items():
    globals()[_name] = _wrap(_name, _target)

__all__ = [
    "figure", "gcf", "gca", "subplots", "subplot", "close", "savefig", "show", "suptitle",
    "colorbar", *_AXES_FUNCS, *_RENAMED,
]
