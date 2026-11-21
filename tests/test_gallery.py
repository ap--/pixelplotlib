"""Smoke-test every README gallery example and style demo."""

import importlib.util
from pathlib import Path

import pytest

import pixelplotlib as ppl

_spec = importlib.util.spec_from_file_location(
    "gallery", Path(__file__).resolve().parents[1] / "examples" / "gallery.py"
)
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)

EXAMPLES = {name: func for name, _, func in gallery.EXAMPLES}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_gallery_example_renders(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if name == "animation":
        (tmp_path / "docs" / "images").mkdir(parents=True)
        EXAMPLES[name]()
        assert (tmp_path / "docs" / "images" / "animation.gif").exists()
        return
    fig = EXAMPLES[name]()
    img = fig.to_array()
    assert img.shape == (fig.height, fig.width, 3)
    assert (img != img[0, 0]).any()


@pytest.mark.parametrize("style", ppl.available())
def test_style_demo_renders(style):
    fig = gallery.style_demo(style)
    assert fig.to_array().shape[:2] == ppl.get_style(style).size[::-1]


def test_readme_gallery_is_up_to_date():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    gallery_md, styles_md = gallery.readme_sections()
    assert gallery_md in readme
    assert styles_md in readme
