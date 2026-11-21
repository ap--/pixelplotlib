"""Visual themes ("styles") that emulate retro hardware."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import palettes
from .colormaps import CmapLike, Ramp, build_ramp
from .colors import ColorLike, ColorSystem, shade
from .fonts import BitmapFont, get_font

Fill = ColorLike | tuple[ColorLike, ColorLike] | None


@dataclass(frozen=True)
class Style:
    """A complete description of a pixel-art look.

    Colors may be anything accepted by :func:`pixelplotlib.colors.parse_color`.
    Fields named ``*_fill`` accept either a color or a ``(top, bottom)`` tuple
    for a dithered vertical gradient.
    """

    name: str
    size: tuple[int, int] = (256, 224)
    scale: int = 3
    palette: Sequence[ColorLike] | None = None
    bits: int | None = None

    background: Fill = "#000000"
    foreground: ColorLike = "#FFFFFF"
    series: Sequence[ColorLike] = ("red", "blue", "green", "yellow")
    linestyles: Sequence[str] = ("-",)
    markers: Sequence[str] = ("o",)
    fill_alphas: Sequence[float] = (1.0,)

    panel: bool = False
    panel_fill: Fill = None
    panel_border: Sequence[ColorLike] = ()
    panel_radius: int = 0
    panel_shadow: ColorLike | None = None
    plot_fill: Fill = None
    legend_fill: Fill = None

    frame_color: ColorLike | None = None
    grid_color: ColorLike | None = None
    grid_linestyle: str = ":"
    text_shadow: ColorLike | None = None
    title_color: ColorLike | None = None
    uppercase: bool = False

    font: str | BitmapFont = "5x7"
    tick_font: str | BitmapFont = "5x7"

    line_width: int = 1
    line_shadow: ColorLike | None = None
    line_halo: ColorLike | None = None
    marker_outline: ColorLike | None = None
    bar_shading: bool = False
    bar_outline: str | None = "auto"

    cmap: CmapLike = "viridis"
    cmap_levels: int = 16
    gradient_levels: int = 8
    dither: bool = True

    pad: int = 2
    tick_length: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def replace(self, **changes) -> Style:
        """Return a copy of this style with some fields changed."""
        return replace(self, **changes)

    @property
    def colors(self) -> ColorSystem:
        cache = self.__dict__.get("_colors")
        if cache is None:
            cache = ColorSystem(self.palette, self.bits)
            object.__setattr__(self, "_colors", cache)
        return cache

    def color(self, c: ColorLike | None, default: ColorLike | None = None):
        """Resolve a user color to a displayable RGB tuple (``None`` stays ``None``)."""
        if c is None:
            if default is None:
                return None
            c = default
        if isinstance(c, str):
            s = c.strip()
            if len(s) >= 2 and s[0] == "C" and s[1:].isdigit():
                return self.series_color(int(s[1:]))
            low = s.lower()
            if low in ("fg", "foreground"):
                return self.color(self.foreground)
            if low in ("bg", "background"):
                use_panel = self.panel and self.panel_fill is not None
                bg = self.panel_fill if use_panel else self.background
                return self.color(bg[0] if isinstance(bg, tuple) else bg)
        if isinstance(c, (int, np.integer)) and self.palette is not None:
            return self.colors.snap(self.palette[int(c) % len(self.palette)])
        return self.colors.snap(c)

    @property
    def fg(self):
        return self.color(self.foreground)

    @property
    def text_color(self):
        return self.fg

    def series_color(self, i: int):
        return self.color(self.series[i % len(self.series)])

    def shade(self, rgb, amount: float):
        """Lighter/darker version of ``rgb`` that is still displayable."""
        return self.colors.snap(shade(rgb, amount))

    def ramp(self, cmap: CmapLike | None = None, levels: int | None = None) -> Ramp:
        return build_ramp(cmap if cmap is not None else self.cmap, self.colors,
                          levels or self.cmap_levels)

    def get_font(self):
        return get_font(self.font)

    def get_tick_font(self):
        return get_font(self.tick_font)

    def text(self, s) -> str:
        s = "" if s is None else str(s)
        return s.upper() if self.uppercase else s


NES_STYLE = Style(
    name="8bit",
    size=(256, 224),
    scale=3,
    palette=palettes.NES,
    background="#000000",
    foreground="#FCFCFC",
    series=("#F83800", "#3CBCFC", "#B8F818", "#F8B800", "#F878F8", "#00E8D8", "#FCA044",
            "#9878F8"),
    markers=("o", "s", "^", "D", "x", "+", "heart", "star"),
    panel=True,
    panel_fill="#000000",
    panel_border=("#FCFCFC",),
    panel_radius=1,
    frame_color="#FCFCFC",
    grid_color="#7C7C7C",
    title_color="#F8B800",
    bar_outline="#000000",
    cmap="viridis",
    cmap_levels=9,
)

SNES_STYLE = Style(
    name="16bit",
    size=(320, 224),
    scale=3,
    bits=5,
    background=("#183080", "#000010"),
    foreground="#F8F8F8",
    series=("#F85858", "#58C8F8", "#78F070", "#F8D030", "#F898F8", "#F8A040", "#B8F858",
            "#C8A0F8"),
    markers=("o", "D", "s", "^", "*", "v", "heart", "star"),
    panel=True,
    panel_fill=("#4868D8", "#101850"),
    panel_border=("#000010", "#F8F8F8", "#A8A8C0"),
    panel_radius=3,
    panel_shadow="#000008",
    legend_fill=("#384898", "#101840"),
    frame_color="#E0E0F0",
    grid_color="#7890E8",
    text_shadow="#000020",
    title_color="#F8E070",
    line_shadow="#000028",
    marker_outline="#000020",
    bar_shading=True,
    cmap="viridis",
    cmap_levels=16,
    gradient_levels=12,
)

GAMEBOY_STYLE = Style(
    name="gameboy",
    size=(160, 144),
    scale=4,
    palette=palettes.GAMEBOY,
    background="#9BBC0F",
    foreground="#0F380F",
    series=("#0F380F", "#306230"),
    linestyles=("-", "-", "--", "--", ":", ":"),
    markers=("o", "s", "^", "x", "D", "+"),
    fill_alphas=(1.0, 0.5, 0.75, 0.25),
    line_halo="bg",
    frame_color="#0F380F",
    grid_color="#8BAC0F",
    grid_linestyle=":",
    bar_outline="#0F380F",
    cmap="gameboy",
    tick_font="3x5",
)

PICO8_STYLE = Style(
    name="pico8",
    size=(192, 160),
    scale=4,
    palette=palettes.PICO8,
    background="#000000",
    foreground="#FFF1E8",
    series=("#FF004D", "#29ADFF", "#00E436", "#FFEC27", "#FF77A8", "#FFA300", "#83769C",
            "#FFCCAA"),
    markers=("o", "s", "^", "D", "x", "+"),
    plot_fill="#1D2B53",
    frame_color="#C2C3C7",
    grid_color="#5F574F",
    title_color="#FFEC27",
    bar_outline="auto",
    cmap="pico",
    font="3x5",
    tick_font="3x5",
)

C64_STYLE = Style(
    name="c64",
    size=(320, 200),
    scale=3,
    palette=palettes.C64,
    background="#6C5EB5",
    foreground="#6C5EB5",
    series=("#FFFFFF", "#B8C76F", "#9AD284", "#9A6759", "#70A4B2", "#6F4F25", "#6F3D86",
            "#959595"),
    markers=("o", "s", "^", "D", "x", "+"),
    panel=True,
    panel_fill="#352879",
    frame_color="#6C5EB5",
    grid_color="#6C5EB5",
    title_color="#FFFFFF",
    uppercase=True,
    bar_outline=None,
    cmap="viridis",
    cmap_levels=9,
)

CGA_STYLE = Style(
    name="cga",
    size=(320, 200),
    scale=3,
    palette=palettes.CGA,
    background="#000000",
    foreground="#FFFFFF",
    series=("#55FFFF", "#FF55FF", "#FFFFFF"),
    linestyles=("-", "-", "-", "--", "--", "--"),
    markers=("o", "s", "^", "x", "D", "+"),
    frame_color="#FFFFFF",
    grid_color="#55FFFF",
    title_color="#FF55FF",
    bar_outline="#000000",
    cmap="gray",
)

ONEBIT_STYLE = Style(
    name="1bit",
    size=(256, 192),
    scale=3,
    palette=palettes.ONE_BIT,
    background="#FFFFFF",
    foreground="#000000",
    series=("#000000",),
    linestyles=("-", "--", ":", "-."),
    markers=("o", "s", "^", "x", "D", "+"),
    fill_alphas=(0.5, 0.25, 0.75, 1.0),
    line_halo="bg",
    marker_outline="bg",
    panel=True,
    panel_fill="#FFFFFF",
    panel_border=("#000000",),
    panel_shadow="#000000",
    frame_color="#000000",
    grid_color="#000000",
    bar_outline="#000000",
    cmap="gray",
)

STYLES: dict[str, Style] = {
    "8bit": NES_STYLE,
    "nes": NES_STYLE,
    "16bit": SNES_STYLE,
    "snes": SNES_STYLE,
    "gameboy": GAMEBOY_STYLE,
    "gb": GAMEBOY_STYLE,
    "pico8": PICO8_STYLE,
    "c64": C64_STYLE,
    "cga": CGA_STYLE,
    "1bit": ONEBIT_STYLE,
    "mac": ONEBIT_STYLE,
}

_current: list[Style] = [NES_STYLE]


def get_style(style: str | Style | None = None) -> Style:
    """Return a :class:`Style` by name (or the current default when ``None``)."""
    if style is None:
        return _current[-1]
    if isinstance(style, Style):
        return style
    try:
        return STYLES[style.lower()]
    except KeyError:
        raise ValueError(f"unknown style {style!r}; choose from {available()}") from None


def available() -> list[str]:
    """Names of the built-in styles (without aliases)."""
    seen, names = set(), []
    for name, st in STYLES.items():
        if id(st) not in seen:
            seen.add(id(st))
            names.append(name)
    return names


def use(style: str | Style) -> None:
    """Set the default style for new figures."""
    _current[-1] = get_style(style)


def register(style: Style, name: str | None = None) -> None:
    """Register a custom style under ``name`` (defaults to ``style.name``)."""
    STYLES[(name or style.name).lower()] = style


@contextmanager
def context(style: str | Style) -> Iterator[Style]:
    """Temporarily use a different default style."""
    _current.append(get_style(style))
    try:
        yield _current[-1]
    finally:
        _current.pop()
