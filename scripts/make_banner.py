"""Draw the README banner, ``assets/banner.png``, from code.

The banner is pixel art: a dark gridded background with corner markers, a cream card holding the
project icon, a giant zoomed crop of the same icon bleeding off the right edge, and the project name
and descriptor set in a hand-made bitmap font. Everything (icon, lettering, grid) is defined in this
file and drawn with Pillow, so the output is reproducible byte for byte and no font file is loaded.

The icon is an hourglass whose falling sand settles into three rising bars: time series, tuned.

Pillow is not a project dependency. Run from the repository root:

    uv run --with pillow==13.0.0 python scripts/make_banner.py

The Pillow version is pinned because the PNG bytes, though not the pixels, can vary between versions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "assets" / "banner.png"

W, H = 1400, 672

Colour = tuple[int, int, int]


def rgb(hex_colour: str) -> Colour:
    return int(hex_colour[1:3], 16), int(hex_colour[3:5], 16), int(hex_colour[5:7], 16)


# Palette: near-black navy background, cream, four blues (navy, blue, sky, pale), white, peach, orange.
BG_DARK = rgb("#0e0e18")
BG_LIGHT = rgb("#1c1b2a")
GRID = rgb("#1f1f2c")
ARC = rgb("#3a3a4a")
MARKER = rgb("#f3f4f0")
CARD = rgb("#f3f2ed")
NAVY = rgb("#04419a")
BLUE = rgb("#0173e4")
SKY = rgb("#44aafb")
PALE = rgb("#b6d8fe")
WHITE = rgb("#ffffff")
PEACH = rgb("#feb092")
ORANGE = rgb("#fa470a")
GLOW = rgb("#ff6a1e")
TEXT = rgb("#f3f2ed")
TEXT_SOFT = rgb("#b6d8fe")
TEXT_MUTED = rgb("#a3adc6")

# ---------------------------------------------------------------------------------------------
# Icon: 14 x 18 cells. A framed hourglass: plates top and bottom, a lit post on the left and a
# shaded one on the right, two glass bulbs joined by a two-cell neck. A little sand is left in the
# top bulb, dipping where it drains into the stream; the rest has fallen and settled into three
# bars that rise from left to right. No columns or parts sit outside the frame.
# ---------------------------------------------------------------------------------------------
ICON = [
    "BBBBBBBBBBBBBB",
    "BSSSSSSSSSSSSB",
    ".S.PPPPPPPP.B.",
    ".SPWWPPPPPPPB.",
    ".SPWPPPPPPPPB.",
    ".S.PPPPPWWP.B.",
    ".S..ooPooo..B.",
    ".S...oOoo...B.",
    ".S....OP....B.",
    ".S....OP....B.",
    ".S...POPP...B.",
    ".S..PPPPPP..B.",
    ".S.PPPPPPOO.B.",
    ".SWPPPOOPOOPB.",
    ".SPOOPOOPOOPB.",
    ".S.OOOOOOOO.B.",
    "BSSSSSSSSSSSSB",
    "NNNNNNNNNNNNNN",
]
ICON_COLOURS = {
    "N": NAVY,
    "B": BLUE,
    "S": SKY,
    "P": PALE,
    "W": WHITE,
    "o": PEACH,
    "O": ORANGE,
}

# ---------------------------------------------------------------------------------------------
# Bitmap font. Each glyph is nine rows separated by "|": rows 0-1 ascenders, rows 2-6 x-height,
# rows 7-8 descenders. "#" is ink. Widths vary per glyph; letters are spaced one cell apart.
# ---------------------------------------------------------------------------------------------
GLYPHS = {
    "a": "....|....|###.|...#|.###|#..#|.###|....|....",
    "b": "#...|#...|###.|#..#|#..#|#..#|###.|....|....",
    "c": "...|...|.##|#..|#..|#..|.##|...|...",
    "d": "...#|...#|.###|#..#|#..#|#..#|.###|....|....",
    "e": "....|....|.##.|#..#|####|#...|.###|....|....",
    "f": ".##|#..|###|#..|#..|#..|#..|...|...",
    "g": "....|....|.###|#..#|#..#|#..#|.###|...#|###.",
    "h": "#...|#...|###.|#..#|#..#|#..#|#..#|....|....",
    "i": "#|.|#|#|#|#|#|.|.",
    "l": "#.|#.|#.|#.|#.|#.|.#|..|..",
    "m": ".....|.....|####.|#.#.#|#.#.#|#.#.#|#.#.#|.....|.....",
    "n": "....|....|###.|#..#|#..#|#..#|#..#|....|....",
    "o": "....|....|.##.|#..#|#..#|#..#|.##.|....|....",
    "p": "....|....|###.|#..#|#..#|#..#|###.|#...|#...",
    "r": "...|...|#.#|##.|#..|#..|#..|...|...",
    "s": "....|....|.###|#...|.##.|...#|###.|....|....",
    "t": "...|#..|###|#..|#..|#..|.##|...|...",
    "u": "....|....|#..#|#..#|#..#|#..#|.###|....|....",
    "v": ".....|.....|#...#|#...#|.#.#.|.#.#.|..#..|.....|.....",
    "y": "....|....|#..#|#..#|#..#|#..#|.###|...#|###.",
    "A": ".##.|#..#|#..#|####|#..#|#..#|#..#|....|....",
    "L": "#...|#...|#...|#...|#...|#...|####|....|....",
    "P": "###.|#..#|#..#|###.|#...|#...|#...|....|....",
    "R": "###.|#..#|#..#|###.|#.#.|#..#|#..#|....|....",
    "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..|.....|.....",
    "0": ".##.|#..#|#..#|#..#|#..#|#..#|.##.|....|....",
    "2": ".##.|#..#|...#|..#.|.#..|#...|####|....|....",
    "-": "...|...|...|...|###|...|...|...|...",
    ".": ".|.|.|.|.|.|#|.|.",
    "·": "..|..|..|##|##|..|..|..|..",
    " ": "..|..|..|..|..|..|..|..|..",
}
GLYPH_ROWS = 9


def glyph(char: str) -> list[str]:
    if char not in GLYPHS:
        raise KeyError(f"no glyph for {char!r}; add one to GLYPHS")
    rows = GLYPHS[char].split("|")
    assert len(rows) == GLYPH_ROWS and len({len(r) for r in rows}) == 1, char
    return rows


def text_width(text: str, scale: int) -> int:
    cells = sum(len(glyph(c)[0]) for c in text) + len(text) - 1
    return cells * scale


def draw_text(img: Image.Image, x: int, y: int, runs: list[tuple[str, Colour]], scale: int) -> None:
    """Draw coloured runs of text with the top of the ascender zone at ``y``."""
    draw = ImageDraw.Draw(img)
    for text, colour in runs:
        for char in text:
            rows = glyph(char)
            for cy, row in enumerate(rows):
                for cx, cell in enumerate(row):
                    if cell == "#":
                        x0, y0 = x + cx * scale, y + cy * scale
                        draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=colour)
            x += (len(rows[0]) + 1) * scale


def draw_icon(img: Image.Image, x: int, y: int, cell: int) -> Image.Image:
    """Draw the icon with its top-left cell corner at (x, y).

    Returns a mask of the orange cells, used to put a soft glow on the giant crop.
    """
    draw = ImageDraw.Draw(img)
    orange = Image.new("L", img.size, 0)
    orange_draw = ImageDraw.Draw(orange)
    for cy, row in enumerate(ICON):
        for cx, key in enumerate(row):
            if key == ".":
                continue
            box = (x + cx * cell, y + cy * cell, x + (cx + 1) * cell - 1, y + (cy + 1) * cell - 1)
            draw.rectangle(box, fill=ICON_COLOURS[key])
            if key == "O":
                orange_draw.rectangle(box, fill=255)
    return orange


def lerp(a: Colour, b: Colour, t: float) -> Colour:
    return round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t), round(a[2] + (b[2] - a[2]) * t)


def background() -> Image.Image:
    """Near-black navy, slightly lighter towards the bottom left, with a soft diagonal sheen."""
    small_w, small_h = W // 4, H // 4
    small = Image.new("RGB", (small_w, small_h))
    pixels = []
    for j in range(small_h):
        for i in range(small_w):
            u, v = i / (small_w - 1), j / (small_h - 1)
            t = max(0.0, v - 0.55 * u) ** 1.4
            # A broad soft band running from the lower left towards the upper middle.
            d = abs((u - 0.05) * 0.9 + (v - 1.0) * -0.55 - 0.35)
            t += 0.35 * max(0.0, 1.0 - d / 0.18) ** 2
            pixels.append(lerp(BG_DARK, BG_LIGHT, min(1.0, t)))
    small.putdata(pixels)
    return small.resize((W, H), Image.BICUBIC)


def arc_layer(centre: tuple[float, float], radius: float) -> Image.Image:
    """An anti-aliased 1 px circle outline as an L mask, drawn at 4x and reduced."""
    ss = 4
    big = Image.new("L", (W * ss, H * ss), 0)
    cx, cy = centre[0] * ss, centre[1] * ss
    r = radius * ss
    ImageDraw.Draw(big).ellipse((cx - r, cy - r, cx + r, cy + r), outline=255, width=ss)
    return big.resize((W, H), Image.BOX)


def marker(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 6) -> None:
    h = size // 2
    draw.rectangle((x - h, y - h, x - h + size - 1, y - h + size - 1), fill=MARKER)


def build() -> Image.Image:
    img = background()

    # --- layout -------------------------------------------------------------------------------
    zoom = 56  # size of one icon cell in the giant crop
    card = 6 * zoom  # the card is one grid cell; a whole number of crop cells keeps edges clean
    card_y = (H - card) // 2
    assert card_y % zoom == 0, "crop rows should line up with the card and the canvas edges"
    cell = 9  # size of one icon cell on the card
    grid_left, text_x = 56, 96

    title_scale, desc_scale, small_scale = 8, 4, 3
    title = [("timestral", TEXT), ("-tune", ORANGE)]
    desc = ["fine-tuning · selective LoRA", "conformal calibration"]
    target = "for Timestral time-series models"
    footer = [("PyTorch ", TEXT_MUTED), ("·", ORANGE), (" Apache-2.0", TEXT_MUTED)]

    widths = [
        text_width("".join(t for t, _ in title), title_scale),
        *(text_width(line, desc_scale) for line in desc),
        text_width(target, small_scale),
        text_width("".join(t for t, _ in footer), small_scale),
    ]
    text_right = text_x + max(widths)
    card_x = text_right + 72
    assert W - (card_x + card) >= 7 * zoom, "card leaves too little room for the zoomed crop"

    # --- grid ---------------------------------------------------------------------------------
    draw = ImageDraw.Draw(img)
    ys = [card_y, card_y + card]
    xs = [grid_left, card_x, card_x + card, card_x + 2 * card]
    for y in ys:
        draw.line((0, y, W, y), fill=GRID)
    for x in xs:
        draw.line((x, 0, x, H), fill=GRID)
    # A faint arc of radius one grid cell through the top-left marker.
    img.paste(ARC, mask=arc_layer((grid_left - 0.75 * card, card_y - 0.67 * card), card))

    # --- giant crop of the icon, bleeding off the right, top and bottom ------------------------
    # A straight zoom of the card's icon, cropped the way a magnifier would: the left post sits
    # behind the card (visible above and below it), column 3 starts at the card's right edge so the
    # neck lands in the middle of the open space, and row 8 (the top of the neck) starts at the
    # card's top edge. That shows the end of the sand in the top bulb, the neck, the rising bars and
    # the base, with the right half of the glass running off the edge.
    origin_x, origin_y = card_x + card - 3 * zoom, card_y - 8 * zoom
    orange = draw_icon(img, origin_x, origin_y, zoom)
    # A slight warm glow on the big orange blocks, centred on the bars.
    glow_size = 8 * zoom
    glow = ImageChops.invert(Image.radial_gradient("L").resize((glow_size, glow_size), Image.BICUBIC))
    glow_mask = Image.new("L", (W, H), 0)
    glow_mask.paste(
        glow.point(lambda v: v * 100 // 255),
        (origin_x + 7 * zoom - glow_size // 2, origin_y + 14 * zoom - glow_size // 2),
    )
    img.paste(GLOW, mask=ImageChops.multiply(glow_mask, orange))

    # --- card with the icon, then markers on top -------------------------------------------------
    draw = ImageDraw.Draw(img)
    draw.rectangle((card_x, card_y, card_x + card - 1, card_y + card - 1), fill=CARD)
    icon_w, icon_h = len(ICON[0]) * cell, len(ICON) * cell
    draw_icon(img, card_x + (card - icon_w) // 2, card_y + (card - icon_h) // 2, cell)
    draw = ImageDraw.Draw(img)
    for x in xs:
        for y in ys:
            marker(draw, x, y)

    # --- lettering, inside the grid cell to the left of the card ---------------------------------
    y = card_y + 36
    draw_text(img, text_x, y, title, title_scale)
    y += 7 * title_scale + 30
    for line in desc:
        runs: list[tuple[str, Colour]] = []
        for i, part in enumerate(line.split("·")):
            if i:
                runs.append(("·", ORANGE))
            runs.append((part, TEXT_SOFT))
        draw_text(img, text_x, y, runs, desc_scale)
        y += 7 * desc_scale + 16
    y += 6
    draw_text(img, text_x, y, [(target, TEXT_MUTED)], small_scale)
    target_bottom = y + 9 * small_scale
    y = card_y + card - 36 - 7 * small_scale
    assert y - target_bottom >= 12, "lettering does not fit in the grid cell"
    draw_text(img, text_x, y, footer, small_scale)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="output path (default: assets/banner.png)")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build().save(args.out, format="PNG", optimize=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
