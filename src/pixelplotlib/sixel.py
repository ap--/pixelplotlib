"""Sixel encoding for showing figures directly in a terminal.

Sixel is a DEC bitmap format understood by terminals such as xterm (``-ti vt340``),
foot, WezTerm, mlterm, Konsole, iTerm2, Windows Terminal and tmux (3.4+). Each
"sixel" character encodes a column of six vertical pixels for one colour register.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import TextIO

import numpy as np

__all__ = ["encode", "terminal_supports_sixel", "write"]

#: Environment variable that forces sixel output on (``1``) or off (``0``).
ENV_VAR = "PIXELPLOTLIB_SIXEL"

_DA1_RESPONSE = re.compile(rb"\x1b\[\?([\d;]*)c")


def encode(image, max_colors: int = 256) -> str:
    """Encode an ``(H, W, 3|4)`` uint8 array or :class:`PIL.Image.Image` as a sixel string.

    Images with more than ``max_colors`` distinct colours are quantized first (pixel-art
    figures rarely need to be). The alpha channel, if any, is ignored.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"expected an (H, W, 3) or (H, W, 4) image, got shape {rgb.shape}")
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    colors, index = _palette_index(rgb, max_colors)

    # P2=1: pixels not set by any sixel keep the terminal background.
    out = ["\x1bP0;1;0q", f'"1;1;{w};{h}']
    for i, (r, g, b) in enumerate(colors.tolist()):
        out.append(f"#{i};2;{round(r * 100 / 255)};{round(g * 100 / 255)};{round(b * 100 / 255)}")
    for y0 in range(0, h, 6):
        band = index[y0:y0 + 6]
        weights = (1 << np.arange(band.shape[0]))[:, None]
        layers = []
        for c in np.unique(band).tolist():
            bits = ((band == c) * weights).sum(axis=0)
            bits = bits[:np.flatnonzero(bits)[-1] + 1]
            layers.append(f"#{c}{_run_length(bits)}")
        out.append("$".join(layers))
        if y0 + 6 < h:
            out.append("-")
    out.append("\x1b\\")
    return "".join(out)


def _palette_index(rgb: np.ndarray, max_colors: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(colors, index)``: an ``(N, 3)`` palette and per-pixel register numbers."""
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8)
    packed |= rgb[..., 2]
    values, index = np.unique(packed, return_inverse=True)
    if len(values) > max_colors:
        from PIL import Image

        quantized = Image.fromarray(rgb, "RGB").quantize(max_colors, dither=Image.Dither.NONE)
        return _palette_index(np.asarray(quantized.convert("RGB")), max_colors)
    colors = np.stack([(values >> 16) & 255, (values >> 8) & 255, values & 255], axis=1)
    return colors.astype(np.uint8), index.reshape(rgb.shape[:2])


def _run_length(bits: np.ndarray) -> str:
    """Sixel characters for one colour layer, with ``!<n><char>`` run-length compression."""
    starts = np.flatnonzero(np.diff(bits, prepend=-1))
    lengths = np.diff(np.append(starts, len(bits)))
    parts = []
    for value, n in zip(bits[starts].tolist(), lengths.tolist()):
        ch = chr(63 + value)
        parts.append(f"!{n}{ch}" if n > 3 else ch * n)
    return "".join(parts)


def write(image, stream: TextIO | None = None) -> None:
    """Write ``image`` as sixel graphics to ``stream`` (default ``sys.stdout``)."""
    stream = stream if stream is not None else sys.stdout
    stream.write(encode(image) + "\n")
    stream.flush()


def terminal_supports_sixel(stream: TextIO | None = None) -> bool:
    """Best-effort check whether ``stream`` (default ``sys.stdout``) can display sixels.

    The ``PIXELPLOTLIB_SIXEL`` environment variable (``1``/``0``) overrides detection.
    Otherwise ``stream`` must be a TTY whose terminal reports sixel support (attribute
    ``4``) in its reply to a Primary Device Attributes query.
    """
    env = os.environ.get(ENV_VAR)
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no", "off")
    stream = stream if stream is not None else sys.stdout
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    return _query_terminal()


def _parse_da1(response: bytes) -> bool:
    """Whether a Primary Device Attributes reply (``ESC [ ? 62 ; 4 ; ... c``) lists sixel."""
    match = _DA1_RESPONSE.search(response)
    return bool(match) and "4" in match.group(1).decode().split(";")


@lru_cache(maxsize=1)
def _query_terminal(timeout: float = 0.2) -> bool:
    try:
        import select
        import termios
        import tty
    except ImportError:  # Windows: rely on PIXELPLOTLIB_SIXEL
        return False
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return False
    response = b""
    try:
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            os.write(fd, b"\x1b[c")
            while not _DA1_RESPONSE.search(response):
                ready, _, _ = select.select([fd], [], [], timeout)
                if not ready:
                    break
                chunk = os.read(fd, 256)
                if not chunk:
                    break
                response += chunk
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (OSError, termios.error):
        return False
    finally:
        os.close(fd)
    return _parse_da1(response)
