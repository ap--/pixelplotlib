"""Animated GIF export."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Callable, Union

import numpy as np

from .figure import Figure, apply_scanlines

FrameLike = Union[Figure, np.ndarray]


def _frame_array(frame: FrameLike, scale: int) -> np.ndarray:
    if isinstance(frame, Figure):
        return frame.render().to_array(scale)
    arr = np.asarray(frame, dtype=np.uint8)
    if scale > 1:
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
    return arr


def save_gif(frames: Iterable[FrameLike], path: str | os.PathLike, fps: float = 10,
             scale: int | None = None, loop: int = 0, scanlines: bool = False) -> None:
    """Write frames (figures or RGB arrays) to an animated GIF.

    Pixel-art frames use few colors, so a shared exact palette is built whenever
    possible (no lossy quantization).
    """
    from PIL import Image

    frames = list(frames)
    if not frames:
        raise ValueError("no frames to save")
    if scale is None:
        scale = frames[0].scale if isinstance(frames[0], Figure) else 1
    arrays = [_frame_array(f, scale) for f in frames]
    if scanlines and scale >= 2:
        arrays = [apply_scanlines(a, scale) for a in arrays]
    packed = [(a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8)
              | a[..., 2] for a in arrays]
    colors = np.unique(np.concatenate([p.ravel() for p in packed]))
    images = []
    if len(colors) <= 256:
        pal = np.zeros((256, 3), dtype=np.uint8)
        pal[: len(colors), 0] = colors >> 16
        pal[: len(colors), 1] = (colors >> 8) & 0xFF
        pal[: len(colors), 2] = colors & 0xFF
        flat_pal = pal.ravel().tolist()
        for p in packed:
            idx = np.searchsorted(colors, p).astype(np.uint8)
            im = Image.fromarray(idx, "P")
            im.putpalette(flat_pal)
            images.append(im)
    else:
        images = [Image.fromarray(a, "RGB").quantize(256, dither=Image.Dither.NONE)
                  for a in arrays]
    duration = int(round(1000 / fps))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration,
                   loop=loop, optimize=False, disposal=1)


def animate(func: Callable[[int], FrameLike], frames: int | Iterable, path: str | os.PathLike,
            fps: float = 10, scale: int | None = None, loop: int = 0,
            scanlines: bool = False) -> None:
    """Call ``func(frame)`` for every frame and save the results as a GIF.

    ``func`` must return a :class:`Figure` (or an RGB array).
    """
    seq = range(frames) if isinstance(frames, int) else frames
    save_gif((func(i) for i in seq), path, fps=fps, scale=scale, loop=loop,
             scanlines=scanlines)
