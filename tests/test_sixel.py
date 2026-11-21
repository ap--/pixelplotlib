import re

import numpy as np
import pytest

import pixelplotlib as ppl
from pixelplotlib import sixel


def decode(data: str) -> np.ndarray:
    """Minimal sixel decoder (colour registers, ``!`` repeats, ``$`` and ``-``)."""
    assert data.startswith("\x1bP") and data.endswith("\x1b\\")
    body = data[data.index("q") + 1:-2]
    w, h = map(int, re.match(r'"1;1;(\d+);(\d+)', body).groups())
    body = body[re.match(r'"[\d;]*', body).end():]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    palette, color, x, y = {}, None, 0, 0
    for token in re.finditer(r"#(\d+)(?:;2;(\d+);(\d+);(\d+))?|!(\d+)(.)|([?-~])|(\$)|(-)", body):
        reg, r, g, b, n, rep, ch, cr, nl = token.groups()
        if reg is not None:
            if r is not None:
                palette[int(reg)] = [round(int(v) * 255 / 100) for v in (r, g, b)]
            else:
                color = palette[int(reg)]
            continue
        if cr:
            x = 0
            continue
        if nl:
            x, y = 0, y + 6
            continue
        count, ch = (int(n), rep) if n else (1, ch)
        bits = ord(ch) - 63
        for dy in range(6):
            if bits >> dy & 1:
                img[y + dy, x:x + count] = color
        x += count
    return img


def test_encode_round_trips_a_figure():
    fig, ax = ppl.subplots(style="pico8")
    ax.plot([0, 1, 2], [1, 0, 1], marker="heart")
    img = fig.to_array(scale=1)  # 160 rows: last sixel band is only partially used
    # sixel colours are RGB percentages, so channels may be off by one or two
    np.testing.assert_allclose(decode(sixel.encode(img)), img, atol=2)


def test_encode_uses_run_length_compression():
    img = np.zeros((6, 20, 3), dtype=np.uint8)
    img[:, :2] = 255
    assert sixel.encode(img) == '\x1bP0;1;0q"1;1;20;6#0;2;0;0;0#1;2;100;100;100#0??!18~$#1~~\x1b\\'


def test_encode_quantizes_to_max_colors():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (12, 40, 3), dtype=np.uint8)
    data = sixel.encode(img, max_colors=16)
    assert len(re.findall(r"#\d+;2;", data)) <= 16
    decoded = decode(data)
    assert decoded.shape == img.shape
    assert len(np.unique(decoded.reshape(-1, 3), axis=0)) <= 16


@pytest.mark.parametrize(
    "response, expected",
    [(b"\x1b[?62;4;6;22c", True), (b"\x1b[?64;1;2;6;22c", False), (b"", False)],
)
def test_parse_device_attributes(response, expected):
    assert sixel._parse_da1(response) == expected


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("off", False)])
def test_env_var_overrides_detection(monkeypatch, value, expected):
    monkeypatch.setenv(sixel.ENV_VAR, value)
    assert sixel.terminal_supports_sixel() is expected


def test_show_auto_writes_sixel_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv(sixel.ENV_VAR, "1")
    fig, ax = ppl.subplots(style="gameboy")
    ax.plot([0, 1], [0, 1])
    fig.show(scale=2)
    assert capsys.readouterr().out == fig.to_sixel(scale=2) + "\n"


def test_show_auto_skips_sixel_when_not_a_tty(monkeypatch, capsys):
    monkeypatch.delenv(sixel.ENV_VAR, raising=False)
    shown = []
    monkeypatch.setattr("PIL.Image.Image.show", lambda self: shown.append(self.size))
    fig, _ = ppl.subplots(style="gameboy")
    fig.show(scale=1)
    assert shown == [(160, 144)]
    assert capsys.readouterr().out == ""
