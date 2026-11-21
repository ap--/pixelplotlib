"""Retro hardware palettes.

All palettes are lists of ``(r, g, b)`` tuples with 8-bit channels.
"""

from __future__ import annotations


def _hex(*codes: str) -> list[tuple[int, int, int]]:
    out = []
    for code in codes:
        code = code.lstrip("#")
        out.append((int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)))
    return out


# Commonly used NES (2C02) palette approximation; the duplicate blacks are removed.
NES = _hex(
    "7C7C7C", "0000FC", "0000BC", "4428BC", "940084", "A80020", "A81000", "881400",
    "503000", "007800", "006800", "005800", "004058", "000000",
    "BCBCBC", "0078F8", "0058F8", "6844FC", "D800CC", "E40058", "F83800", "E45C10",
    "AC7C00", "00B800", "00A800", "00A844", "008888",
    "F8F8F8", "3CBCFC", "6888FC", "9878F8", "F878F8", "F85898", "F87858", "FCA044",
    "F8B800", "B8F818", "58D854", "58F898", "00E8D8", "787878",
    "FCFCFC", "A4E4FC", "B8B8F8", "D8B8F8", "F8B8F8", "F8A4C0", "F0D0B0", "FCE0A8",
    "F8D878", "D8F878", "B8F8B8", "B8F8D8", "00FCFC", "F8D8F8",
)

GAMEBOY = _hex("0F380F", "306230", "8BAC0F", "9BBC0F")

PICO8 = _hex(
    "000000", "1D2B53", "7E2553", "008751", "AB5236", "5F574F", "C2C3C7", "FFF1E8",
    "FF004D", "FFA300", "FFEC27", "00E436", "29ADFF", "83769C", "FF77A8", "FFCCAA",
)

# "Pepto" Commodore 64 palette.
C64 = _hex(
    "000000", "FFFFFF", "68372B", "70A4B2", "6F3D86", "588D43", "352879", "B8C76F",
    "6F4F25", "433900", "9A6759", "444444", "6C6C6C", "9AD284", "6C5EB5", "959595",
)

# IBM CGA mode 4, palette 1 (high intensity).
CGA = _hex("000000", "55FFFF", "FF55FF", "FFFFFF")

ONE_BIT = _hex("000000", "FFFFFF")

PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "nes": NES,
    "gameboy": GAMEBOY,
    "pico8": PICO8,
    "c64": C64,
    "cga": CGA,
    "1bit": ONE_BIT,
}


def get_palette(name: str) -> list[tuple[int, int, int]]:
    """Return a copy of a named palette."""
    try:
        return list(PALETTES[name.lower()])
    except KeyError:
        raise ValueError(f"unknown palette {name!r}; choose from {sorted(PALETTES)}") from None
