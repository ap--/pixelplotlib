"""Render the README gallery.

Run with ``uv run python examples/gallery.py``. Every example function below is
rendered to ``docs/images/<name>.png`` (or ``.gif``) and its source code is copied
into the README between the ``<!-- gallery:* -->`` / ``<!-- styles:* -->`` markers,
so the documentation always shows exactly the code that produced each image.
"""

from __future__ import annotations

import argparse
import inspect
import re
import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np

import pixelplotlib as ppl

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
README = ROOT / "README.md"

EXAMPLES: list[tuple[str, str, Callable]] = []


def example(title: str):
    def register(func):
        EXAMPLES.append((func.__name__, title, func))
        return func

    return register


# --------------------------------------------------------------------------- plot types
@example("Line plot")
def line_plot():
    x = np.linspace(0, 4 * np.pi, 200)
    fig, ax = ppl.subplots(style="16bit")
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, np.cos(x), label="cos(x)")
    ax.plot(x[::12], np.sin(x[::12]) * np.cos(x[::12]), "o--", label="product")
    ax.set(title="TRIG FUNCTIONS", xlabel="x [rad]", ylabel="amplitude")
    ax.legend(loc="lower left")
    return fig


@example("Scatter plot")
def scatter_plot():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, 150))
    fig, ax = ppl.subplots(style="8bit")
    sc = ax.scatter(x, y, c=np.hypot(x, y), cmap="fire")
    ax.set(title="RANDOM ENCOUNTERS", xlabel="x", ylabel="y")
    ax.colorbar(sc, label="r")
    return fig


@example("Sprite markers")
def sprite_markers():
    fig, ax = ppl.subplots(style="pico8")
    markers = ["heart", "star", "skull", "ufo", "drop", "o", "s", "^", "D", "x"]
    for i, m in enumerate(markers):
        ax.scatter([i % 5], [i // 5], marker=m, size=2, color=f"C{i}")
        ax.text(i % 5, i // 5 - 0.45, m, ha="center", va="top")
    ax.set(title="MARKERS", xlim=(-0.6, 4.6), ylim=(-0.9, 1.5))
    ax.axis("off")
    return fig


@example("Bar chart")
def bar_chart():
    labels = ["Fire", "Ice", "Bolt", "Cure"]
    fig, ax = ppl.subplots(style="16bit")
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [42, 27, 35, 18], width=0.4, label="Lv 10")
    ax.bar(x + 0.2, [65, 48, 51, 33], width=0.4, label="Lv 20")
    ax.set_xticks(x, labels)
    ax.set(title="SPELL DAMAGE", ylabel="HP")
    ax.legend(loc="upper right")
    return fig


@example("Horizontal bars")
def barh_chart():
    fig, ax = ppl.subplots(style="c64")
    langs = ["BASIC", "C", "ASM", "PASCAL", "FORTH"]
    ax.barh(langs, [91, 64, 48, 27, 12], color=[f"C{i}" for i in range(5)])
    ax.set(title="Favourite language", xlabel="votes")
    return fig


@example("Histogram")
def histogram():
    rng = np.random.default_rng(7)
    fig, ax = ppl.subplots(style="gameboy")
    ax.hist(rng.normal(0, 1, 2000), bins=24, label="normal")
    ax.hist(rng.normal(1.5, 0.6, 1000), bins=24, histtype="step", label="shifted")
    ax.set(title="HISTOGRAM", xlabel="value", ylabel="count")
    ax.legend()
    return fig


@example("2-D histogram")
def histogram_2d():
    rng = np.random.default_rng(3)
    x = rng.normal(size=20000)
    y = 0.6 * x + rng.normal(scale=0.7, size=20000)
    fig, ax = ppl.subplots(style="16bit")
    _, _, _, mesh = ax.hist2d(x, y, bins=32, cmap="magma")
    ax.colorbar(mesh, label="counts")
    ax.set(title="DENSITY", xlabel="x", ylabel="y")
    return fig


@example("Filled areas")
def fill_between():
    x = np.linspace(0, 10, 120)
    y = np.sin(x) * np.exp(-x / 8)
    fig, ax = ppl.subplots(style="1bit")
    ax.fill_between(x, y - 0.3, y + 0.3, label="95% band")
    ax.plot(x, y, label="model")
    ax.set(title="Confidence band", xlabel="time", ylabel="signal")
    ax.legend()
    return fig


@example("Error bars")
def error_bars():
    rng = np.random.default_rng(5)
    x = np.arange(1, 9)
    y = 2.0 * x + rng.normal(0, 1.5, x.size)
    fig, ax = ppl.subplots(style="8bit")
    ax.errorbar(x, y, yerr=rng.uniform(1, 3, x.size), label="data")
    ax.plot(x, 2.0 * x, "--", label="fit")
    ax.set(title="MEASUREMENTS", xlabel="dose", ylabel="response")
    ax.legend(loc="upper left")
    return fig


@example("Step and stem plots")
def step_and_stem():
    n = np.arange(16)
    fig, (ax1, ax2) = ppl.subplots(1, 2, style="snes")
    ax1.step(n, np.cumsum(n % 3), where="post")
    ax1.set(title="step", xlabel="n")
    ax2.stem(n, np.sin(n / 2.5))
    ax2.set(title="stem", xlabel="n")
    return fig


@example("Image / heatmap")
def heatmap():
    y, x = np.mgrid[-2:2:64j, -3:3:96j]
    z = np.sin(x * 2) * np.cos(y * 3) * np.exp(-(x**2 + y**2) / 4)
    fig, ax = ppl.subplots(style="16bit")
    im = ax.imshow(z, cmap="coolwarm", extent=(-3, 3, -2, 2), origin="lower")
    ax.colorbar(im)
    ax.set(title="WAVE PACKET", xlabel="x", ylabel="y")
    return fig


@example("Mandelbrot (imshow without dithering)")
def mandelbrot():
    y, x = np.mgrid[-1.2:1.2:150j, -2.1:0.9:200j]
    c = x + 1j * y
    z = np.zeros_like(c)
    count = np.zeros(c.shape)
    for _ in range(40):
        z = np.where(np.abs(z) < 4, z * z + c, z)
        count += np.abs(z) < 4
    fig, ax = ppl.subplots(style="pico8")
    ax.imshow(count, cmap="pico", dither=False, origin="lower", extent=(-2.1, 0.9, -1.2, 1.2))
    ax.set(title="MANDELBROT", xlabel="Re", ylabel="Im")
    return fig


@example("Pixelated RGB image")
def rgb_image():
    y, x = np.mgrid[0:1:96j, 0:1:128j]
    sky = np.stack([40 + 120 * y, 60 + 140 * y, 200 - 60 * y], axis=-1)
    sun = np.hypot(x - 0.7, y - 0.55) < 0.18
    sky[sun] = (255, 200, 60)
    ground = y > 0.75 + 0.05 * np.sin(x * 20)
    sky[ground] = np.stack([30 + 20 * x, 120 + 60 * x, 40 + 0 * x], axis=-1)[ground]
    fig, ax = ppl.subplots(style="gameboy")
    ax.imshow(sky.astype(np.uint8))
    ax.set(title="RGB -> 4 SHADES")
    ax.axis("off")
    return fig


@example("Pcolormesh (irregular grid)")
def pcolormesh():
    xe = np.geomspace(1, 100, 17)
    ye = np.linspace(0, 1, 11)
    c = np.log10(xe[:-1])[None, :] * np.sin(np.pi * ye[:-1])[:, None]
    fig, ax = ppl.subplots(style="c64")
    mesh = ax.pcolormesh(xe, ye, c, cmap="fire")
    ax.set_xscale("log")
    ax.colorbar(mesh)
    ax.set(title="Log-spaced cells", xlabel="frequency")
    return fig


@example("Contour lines")
def contour_lines():
    y, x = np.mgrid[-2:2:80j, -3:3:120j]
    z = np.exp(-((x - 1) ** 2 + y**2)) - np.exp(-((x + 1) ** 2 + (y - 0.5) ** 2))
    fig, ax = ppl.subplots(style="1bit")
    ax.contour(x, y, z, levels=10, colors="fg")
    ax.set(title="Dipole potential", xlabel="x", ylabel="y")
    return fig


@example("Filled contours")
def filled_contours():
    y, x = np.mgrid[-3:3:90j, -3:3:120j]
    z = np.sin(x) ** 2 + np.cos(y + x / 2) + 0.1 * x
    fig, ax = ppl.subplots(style="16bit")
    cs = ax.contourf(x, y, z, levels=8, cmap="terrain")
    ax.contour(x, y, z, levels=8, colors="#000000")
    ax.colorbar(cs)
    ax.set(title="TERRAIN MAP", xlabel="x", ylabel="y")
    return fig


@example("Vector field (quiver)")
def quiver():
    y, x = np.mgrid[-2:2:9j, -2:2:11j]
    u, v = -y, x
    fig, ax = ppl.subplots(style="8bit")
    ax.quiver(x, y, u, v, C=np.hypot(u, v), cmap="rainbow")
    ax.set(title="VORTEX", xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    return fig


@example("Pie chart")
def pie_chart():
    fig, ax = ppl.subplots(style="16bit")
    ax.pie([45, 25, 20, 10], labels=["HP", "MP", "XP", "GP"], autopct="%d%%")
    ax.set_title("INVENTORY")
    return fig


@example("Box plot")
def box_plot():
    rng = np.random.default_rng(11)
    data = [rng.normal(mu, sd, 200) for mu, sd in [(0, 1), (1, 0.5), (-0.5, 1.5), (2, 0.8)]]
    fig, ax = ppl.subplots(style="cga")
    ax.boxplot(data, labels=["A", "B", "C", "D"])
    ax.set(title="BOX PLOT", ylabel="value")
    return fig


@example("Violin plot")
def violin_plot():
    rng = np.random.default_rng(12)
    data = [rng.normal(0, 1, 300), np.concatenate([rng.normal(-1.5, 0.5, 150),
            rng.normal(1.5, 0.5, 150)]), rng.gamma(2, 1, 300) - 2]
    fig, ax = ppl.subplots(style="pico8")
    ax.violinplot(data, labels=["normal", "bimodal", "skewed"])
    ax.set(title="VIOLINS", ylabel="value")
    return fig


@example("Log scales")
def log_scales():
    n = np.geomspace(1, 1e4, 40)
    fig, ax = ppl.subplots(style="gameboy")
    ax.plot(n, n, label="O(n)")
    ax.plot(n, n * np.log2(n), label="O(n log n)")
    ax.plot(n, n**1.5, label="O(n^1.5)")
    ax.set(xscale="log", yscale="log", title="COMPLEXITY", xlabel="n", ylabel="ops")
    ax.legend(loc="upper left")
    return fig


@example("Annotations, reference lines and spans")
def annotations():
    x = np.linspace(0, 10, 200)
    y = np.exp(-((x - 6) ** 2)) + 0.3 * np.exp(-((x - 2.5) ** 2) / 0.5)
    fig, ax = ppl.subplots(style="16bit")
    ax.axvspan(5, 7, color="C2", alpha=0.25)
    ax.axhline(0.5, color="C3", linestyle="--")
    ax.plot(x, y)
    ax.annotate("PEAK!", xy=(6, 1.0), xytext=(8.3, 0.85))
    ax.text(2.5, 0.38, "shoulder", ha="center")
    ax.set(title="SPECTRUM", xlabel="energy", ylabel="intensity", ylim=(0, 1.15))
    return fig


@example("Subplot grid")
def subplot_grid():
    rng = np.random.default_rng(2)
    fig = ppl.Figure(size=(320, 240), style="16bit")
    fig.suptitle("MISSION REPORT")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(np.cumsum(rng.normal(size=100)))
    ax1.set_title("walk")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(["A", "B", "C"], [3, 5, 2])
    ax2.set_title("loot")
    ax3 = fig.add_subplot(2, 2, (3, 4))
    ax3.hist(rng.normal(size=500), bins=30)
    ax3.set_title("damage rolls")
    return fig


@example("Animated GIF")
def animation():
    x = np.linspace(0, 2 * np.pi, 120)

    def frame(i):
        fig, ax = ppl.subplots(style="gameboy")
        ax.plot(x, np.sin(x + i * np.pi / 8))
        ax.scatter([x[i * 6]], [np.sin(x[i * 6] + i * np.pi / 8)], marker="heart", size=2)
        ax.set(title="ANIMATION", ylim=(-1.2, 1.2))
        return fig

    ppl.animate(frame, 16, "docs/images/animation.gif", fps=8)


# --------------------------------------------------------------------------- styles
def style_demo(style):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 100)
    fig, (ax1, ax2) = ppl.subplots(1, 2, style=style)
    ax1.fill_between(x, np.sin(x) - 0.4, np.sin(x) + 0.4)
    ax1.plot(x, np.sin(x), label="sin")
    ax1.plot(x, np.cos(x), label="cos")
    ax1.set(title=style, xlabel="t")
    ax2.hist(rng.normal(size=300), bins=8)
    ax2.set(title="hist")
    return fig


# --------------------------------------------------------------------------- driver
def _body(func) -> str:
    """Source of an example function without its decorator, signature and ``return``."""
    lines = inspect.getsource(func).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def ")) + 1
    body = textwrap.dedent("\n".join(lines[start:]))
    return re.sub(r"\nreturn fig\s*$", "", body).rstrip()


def _replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"README is missing the {start} ... {end} markers")
    return pattern.sub(lambda _: f"{start}\n{content}\n{end}", text)


def render_examples(only: set[str] | None = None) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name, _, func in EXAMPLES:
        if only and name not in only:
            continue
        if name == "animation":
            func()  # writes the GIF itself (path is relative to the repository root)
        else:
            func().save(IMAGES / f"{name}.png")
        print(f"rendered {name}")
    if not only or "styles" in only:
        for style in ppl.available():
            style_demo(style).save(IMAGES / f"style_{style}.png")
            print(f"rendered style_{style}")


def readme_sections() -> tuple[str, str]:
    gallery = []
    for name, title, func in EXAMPLES:
        ext = "gif" if name == "animation" else "png"
        code = _body(func)
        if name != "animation":
            code += f'\nfig.save("{name}.png")'
        gallery += [f"### {title}", "", f"![{title}](docs/images/{name}.{ext})", "",
                    "```python", code, "```", ""]
    rows, cells = [], []
    for style in ppl.available():
        st = ppl.get_style(style)
        w, h = st.size
        cells.append(f'<td align="center"><img src="docs/images/style_{style}.png" width="320">'
                     f"<br><code>{style}</code> ({w}x{h})</td>")
        if len(cells) == 2:
            rows.append("<tr>" + "".join(cells) + "</tr>")
            cells = []
    if cells:
        rows.append("<tr>" + "".join(cells) + "</tr>")
    styles = "<table>\n" + "\n".join(rows) + "\n</table>"
    styles += "\n\n<details><summary>Code for the style demo</summary>\n\n```python\n"
    styles += _body(style_demo).replace("style=style", 'style="16bit"') + "\n```\n\n</details>"
    return "\n".join(gallery).rstrip(), styles


def update_readme() -> None:
    gallery, styles = readme_sections()
    text = README.read_text()
    text = _replace_block(text, "gallery", gallery)
    text = _replace_block(text, "styles", styles)
    README.write_text(text)
    print(f"updated {README.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="only render these examples ('styles' too)")
    parser.add_argument("--no-readme", action="store_true", help="do not rewrite README.md")
    args = parser.parse_args()
    import os

    os.chdir(ROOT)
    render_examples(set(args.names) or None)
    if not args.no_readme:
        update_readme()


if __name__ == "__main__":
    main()
