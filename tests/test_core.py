import numpy as np
import pytest
from PIL import Image

import pixelplotlib as ppl
from pixelplotlib.colors import mix, parse_color, quantize_bits
from pixelplotlib.ticks import format_ticks, nice_ticks


@pytest.mark.parametrize(
    "lo, hi, max_ticks, expected",
    [
        (-0.5, 10.5, 2, [0, 10]),  # used to return only [0]
        (0, 10, 6, [0, 2, 4, 6, 8, 10]),
        (0, 1, 5, [0, 0.25, 0.5, 0.75, 1]),
    ],
)
def test_nice_ticks(lo, hi, max_ticks, expected):
    ticks, _ = nice_ticks(lo, hi, max_ticks)
    np.testing.assert_allclose(ticks, expected)


def test_format_ticks_log_uses_one_notation_for_all_decades():
    labels = format_ticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6], log=True)
    assert labels == ["1E0", "1E1", "1E2", "1E3", "1E4", "1E5", "1E6"]
    assert format_ticks([1, 10, 100], log=True) == ["1", "10", "100"]


def test_mix_uint8_does_not_overflow():
    red = np.array([200, 80, 80], dtype=np.uint8)
    darker = mix(red, (0, 0, 0), 0.45)
    assert all(d < r for d, r in zip(darker, red))


def _row_of(img, rgb):
    ys, _ = np.nonzero(np.all(img == rgb, axis=-1))
    assert ys.size, f"{rgb} not found in image"
    return ys.mean()


@pytest.mark.parametrize(
    "kwargs, top_is_first_row",
    [
        ({}, True),
        ({"origin": "lower"}, False),
        ({"extent": (0, 1, 1, 0)}, True),
        ({"extent": (0, 1, 0, 1)}, True),
        ({"origin": "lower", "extent": (0, 1, 0, 1)}, False),
    ],
)
def test_imshow_orientation(kwargs, top_is_first_row):
    magenta, green = (255, 0, 255), (0, 255, 0)
    data = np.array([[magenta], [green]], dtype=np.uint8)
    fig, ax = ppl.subplots(style="16bit")
    ax.imshow(data, **kwargs)
    img = fig.to_array()
    first_above = _row_of(img, magenta) < _row_of(img, green)
    assert first_above == top_is_first_row


def _busy_figure(style):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    fig, (ax1, ax2) = ppl.subplots(1, 2, style=style)
    ax1.fill_between(x, np.sin(x) - 0.5, np.sin(x) + 0.5)
    ax1.plot(x, np.sin(x), label="sin")
    ax1.scatter(x[::5], np.cos(x[::5]), marker="heart", label="cos")
    ax1.bar([2, 5, 8], [0.5, 1, 0.3], alpha=0.5)
    ax1.legend()
    ax1.set(title="Palette", xlabel="x")
    im = ax2.imshow(rng.random((16, 16)))
    fig.colorbar(im, ax=ax2)
    return fig


@pytest.mark.parametrize("style", ["8bit", "gameboy", "pico8", "c64", "cga", "1bit"])
def test_palette_styles_only_use_palette_colors(style):
    palette = {parse_color(c) for c in ppl.get_style(style).palette}
    img = _busy_figure(style).to_array()
    used = {tuple(int(v) for v in c) for c in np.unique(img.reshape(-1, 3), axis=0)}
    assert used <= palette


def test_16bit_style_uses_5_bit_channels():
    img = _busy_figure("16bit").to_array()
    np.testing.assert_array_equal(quantize_bits(img, 5), img)


def test_save_upscales_with_nearest_neighbour(tmp_path):
    fig, ax = ppl.subplots(style="gameboy")
    ax.plot([0, 1, 2], [1, 0, 1])
    fig.save(tmp_path / "out.png", scale=2)
    native = fig.to_array()
    saved = np.asarray(Image.open(tmp_path / "out.png").convert("RGB"))
    np.testing.assert_array_equal(saved, native.repeat(2, axis=0).repeat(2, axis=1))


def test_animate_writes_gif(tmp_path):
    def frame(i):
        fig, ax = ppl.subplots(style="pico8")
        ax.plot([0, 1], [0, i])
        return fig

    path = tmp_path / "anim.gif"
    ppl.animate(frame, 3, path, fps=5, scale=1)
    with Image.open(path) as gif:
        assert gif.n_frames == 3
        assert gif.size == ppl.get_style("pico8").size
