"""pixelplotlib: scientific plots with 8-bit and 16-bit pixel-art graphics.

Quick start::

    import numpy as np
    import pixelplotlib as ppl

    x = np.linspace(0, 10, 200)
    fig, ax = ppl.subplots(style="16bit")
    ax.plot(x, np.sin(x), label="sin")
    ax.set(title="HELLO WAVES", xlabel="t", ylabel="A")
    ax.legend()
    fig.save("waves.png")
"""

from . import colormaps, colors, palettes, styles
from .animation import animate, save_gif
from .axes import Axes
from .canvas import Canvas
from .colormaps import Colormap, get_cmap, register_cmap
from .dither import pixelate
from .figure import Figure
from .fonts import FONT_3X5, FONT_5X7, BitmapFont
from .markers import MARKERS
from .pyplot import *  # noqa: F401,F403
from .pyplot import __all__ as _pyplot_all
from .styles import Style, available, context, get_style, register, use

__version__ = "0.1.0"

__all__ = [
    "Axes", "BitmapFont", "Canvas", "Colormap", "Figure", "Style", "FONT_3X5", "FONT_5X7",
    "MARKERS", "animate", "available", "colormaps", "colors", "context", "get_cmap",
    "get_style", "palettes", "pixelate", "register", "register_cmap", "save_gif", "styles",
    "use", *_pyplot_all,
]
