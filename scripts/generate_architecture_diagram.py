"""Renders assets/architecture.png from MoviBot's actual modules and data.

Offline only (PIL, no network or model calls).

Two things this diagram must not drift from, so both are read rather than typed:
module names come from agent.tools.TRACE_NAMES and the turn bound from
agent.loop.MAX_ROUNDS. The assignment requires identical naming across the
diagram, the /api/execute steps trace, and agent_info.json, so none of it is
typed here.

There is deliberately no data layer. Drawing one meant either a store per tool,
which is false -- LexicalScreen and PlotSearch read the same passage index --
or a shared bus, which implies every tool reads every store and is equally
false. The counts live in the Architecture tab and /api/rag/info instead.

Usage: python scripts/generate_architecture_diagram.py
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import loop, tools  # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA = os.path.join(_ROOT, "data_preprocessing", "data_ready")
OUTPUT_PATH = os.path.join(_ROOT, "assets", "architecture.png")

WIDTH, HEIGHT = 1140, 420

BG = (255, 255, 255)
INK = (24, 28, 34)
MUTED = (118, 124, 132)
RULE = (226, 230, 235)
ARROW = (128, 134, 142)

# Blue marks the one metered step; green marks everything that is free and
# local. That distinction is the main thing a reader should take away.
PAID_FILL, PAID_LINE = (233, 240, 253), (48, 92, 176)
FREE_FILL, FREE_LINE = (233, 246, 237), (46, 125, 80)
DATA_FILL, DATA_LINE = (248, 249, 250), (176, 183, 191)
PLAIN_FILL, PLAIN_LINE = (255, 255, 255), (150, 157, 165)


def _font(size: int, bold: bool = False):
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Helvetica.ttc"]
        if bold else
        ["/System/Library/Fonts/Helvetica.ttc",
         "/System/Library/Fonts/Supplemental/Arial.ttf"]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


_EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"


def _emoji(char: str, size: int) -> "Image.Image | None":
    """An emoji as an RGBA image, at any size.

    Apple Color Emoji is a bitmap font with fixed strikes -- 20, 32 and 64 are
    valid, 16 raises "invalid pixel size". So it is always rendered at 64 and
    downsampled, which also anti-aliases better than asking for a small strike.
    Returns None if the font is missing, so a Linux box still renders the
    diagram without emoji rather than crashing.
    """
    if not os.path.exists(_EMOJI_FONT):
        return None
    try:
        font = ImageFont.truetype(_EMOJI_FONT, 64)
    except OSError:
        return None
    tile = Image.new("RGBA", (76, 76), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((4, 4), char, font=font, embedded_color=True)
    return tile.crop(tile.getbbox() or (0, 0, 76, 76)).resize(
        (size, size), Image.LANCZOS
    )


def _centre(draw, cx, y, text, font, fill):
    b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (b[2] - b[0]) / 2, y), text, fill=fill, font=font)


def _box(draw, xy, title, lines=(), fill=PLAIN_FILL, line=PLAIN_LINE,
         title_size=22, img=None, emoji=None):
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=line, width=2)
    x0, y0, x1, y1 = xy
    cx = (x0 + x1) / 2

    tf = _font(title_size, bold=True)
    sf = _font(16)
    block = (draw.textbbox((0, 0), title, font=tf)[3] + 4) + len(lines) * 21
    y = (y0 + y1) / 2 - block / 2

    _centre(draw, cx, y, title, tf, INK)
    y += draw.textbbox((0, 0), title, font=tf)[3] + 6
    for text in lines:
        # An emoji cannot be drawn in the same call as the text -- it comes
        # from a different font -- so the pair is centred as one unit and the
        # two are placed side by side.
        glyph = _emoji(emoji, 19) if (emoji and text is lines[0]) else None
        if glyph is not None and img is not None:
            tw = draw.textbbox((0, 0), text, font=sf)[2]
            total = glyph.width + 5 + tw
            gx = int(cx - total / 2)
            img.paste(glyph, (gx, int(y) - 1), glyph)
            draw.text((gx + glyph.width + 5, y), text, fill=MUTED, font=sf)
        else:
            _centre(draw, cx, y, text, sf, MUTED)
        y += 21


def _arrow(draw, p0, p1, both=False, width=2):
    draw.line([p0, p1], fill=ARROW, width=width)

    def head(tip, frm):
        dx, dy = tip[0] - frm[0], tip[1] - frm[1]
        n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / n, dy / n
        s = 8
        draw.polygon([
            tip,
            (tip[0] - s * ux + s * 0.5 * -uy, tip[1] - s * uy + s * 0.5 * ux),
            (tip[0] - s * ux - s * 0.5 * -uy, tip[1] - s * uy - s * 0.5 * ux),
        ], fill=ARROW)

    head(p1, p0)
    if both:
        head(p0, p1)


def _label(draw, cx, cy, text, font=None):
    font = font or _font(15)
    b = draw.textbbox((0, 0), text, font=font)
    w, h = b[2] - b[0], b[3] - b[1]
    draw.rectangle((cx - w / 2 - 5, cy - h / 2 - 3, cx + w / 2 + 5, cy + h / 2 + 5), fill=BG)
    draw.text((cx - w / 2, cy - h / 2 - 1), text, fill=MUTED, font=font)


def _band(draw, y, text, note):
    f, fn = _font(14, bold=True), _font(14)
    draw.text((40, y), text.upper(), fill=MUTED, font=f)
    w = draw.textbbox((0, 0), text.upper(), font=f)[2]
    draw.text((40 + w + 12, y), note, fill=(180, 186, 193), font=fn)


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.text((40, 28), "MoviBot Architecture", fill=INK, font=_font(34, bold=True))

    # ---- request / planner / answer -----------------------------------
    req = (36, 112, 236, 186)
    plan = (400, 100, 740, 198)
    ans = (924, 112, 1104, 186)

    _box(draw, req, "User request")
    _box(draw, plan, tools.TRACE_NAMES.get("planner", "Planner"),
         ("Which tool should I use?",),
         fill=PAID_FILL, line=PAID_LINE, title_size=24,
         img=img, emoji="\U0001F527")
    _box(draw, ans, "Final answer")

    _arrow(draw, (236, 149), (396, 149))
    _arrow(draw, (744, 149), (920, 149))
    _label(draw, 832, 149, "no more tools needed")

    # Right-aligned to stop short of x=640, where the planner's bus line drops
    # to the tool row -- centred under the box, the line struck through it.
    metered = f"metered · at most {loop.MAX_ROUNDS} model turns"
    mf = _font(14)
    draw.text((554 - draw.textbbox((0, 0), metered, font=mf)[2], 203), metered,
              fill=PAID_LINE, font=mf)

    # ---- tools ---------------------------------------------------------
    _band(draw, 244, "Tools", "")

    # Left to right is cheapest to dearest, which is also the order the planner
    # is told to work in. The ordering is the design, so the diagram encodes it.
    # One line each, naming the mechanism rather than explaining it. The
    # explanation lives in the tab under the image; the diagram only has to say
    # what each box *is*.
    # The emoji carry the distinction the words repeat: sorting, excluding,
    # searching, reading. Deliberately four different actions.
    specs = [
        ("filter_catalog", ("SQL-like column filter",), "\U0001F5C2"),
        ("screen_out", ("exhaustive regex scan",), "\U0001F6AB"),
        ("search_plots", ("vector similarity search",), "\U0001F50E"),
        ("read_synopses", ("full plot text",), "\U0001F4D6"),
    ]
    tw, gap = 252, 28
    start = (WIDTH - (tw * len(specs) + gap * (len(specs) - 1))) / 2
    ty0, ty1 = 274, 366

    tool_boxes = []
    for i, (key, sub, glyph) in enumerate(specs):
        x0 = start + i * (tw + gap)
        box = (x0, ty0, x0 + tw, ty1)
        tool_boxes.append(box)
        _box(draw, box, tools.TRACE_NAMES[key], sub, fill=FREE_FILL, line=FREE_LINE,
             title_size=20, img=img, emoji=glyph)

    # One bus off the planner, then a two-way link into each tool: the planner
    # calls, the result comes back to the planner. Drawn as double-headed
    # arrows rather than separate call/return lines, which turned into a
    # thicket of overlapping dashes in the previous version.
    bus_y = 256
    draw.line([(570, 198), (570, bus_y)], fill=ARROW, width=2)
    draw.line([((tool_boxes[0][0] + tool_boxes[0][2]) / 2, bus_y),
               ((tool_boxes[-1][0] + tool_boxes[-1][2]) / 2, bus_y)], fill=ARROW, width=2)
    for box in tool_boxes:
        cx = (box[0] + box[2]) / 2
        _arrow(draw, (cx, bus_y), (cx, ty0 - 2), both=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  modules: {', '.join(tools.TRACE_NAMES[k] for k, _, _ in specs)}")


if __name__ == "__main__":
    main()
