"""Renders assets/architecture.png from MoviBot's actual modules and data.

Offline only (PIL, no network or model calls).

Two things this diagram must not drift from, so both are read rather than typed:
module names come from agent.tools.TRACE_NAMES and the turn bound from
agent.loop.MAX_ROUNDS. The assignment requires identical naming across the
diagram, the /api/execute steps trace, and agent_info.json, so none of it is
typed here.

WHY THE CYCLE IS THE SUBJECT

An earlier version drew the planner fanning out to four tools. Everything in it
was true, but with no return edge and no turn counter it read as one-shot
dispatch -- a router in front of a fixed pipeline. That is the wrong takeaway:
agent/loop.py offers every tool on every turn, the model picks which and how
many, and the loop exits only when it emits no tool call. So the cycle, not the
fan-out, is what the diagram has to show: Reason -> Act -> Observe -> Stop?,
with the "No" edge closing it and the turn bound printed on the frame.

The tools are drawn as an inventory the loop reaches into, deliberately *not*
as a left-to-right chain. There is no code path that runs them in order; the
ordering is a preference stated in the system prompt, and a diagram that chains
them asserts a mechanism that does not exist. The connector leaves the frame as
a whole rather than the Act box, because any turn may call any tool.

WHY THE CANVAS IS NEARLY SQUARE

The Architecture tab renders this inside a 900px container with `max-width:
100%`. A wide canvas is therefore downscaled hard -- the previous 1240x560
landscape lost a third of its size before it reached the reader, which took
15px labels down to about 10px on screen. So the layout is laid out portrait-
ish, close to the container width, and the whole thing is drawn at SS times
final size and resampled down. Nothing here is sized for a slide; it is sized
for that column.

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
OUTPUT_PATH = os.path.join(_ROOT, "assets", "architecture.png")

# Every coordinate below is in layout units, which are the pixels the reader
# actually gets. SS only buys resampling quality: the image is drawn SS times
# larger and shrunk back down at save time, so curves and small type stay clean
# on a retina display and survive the browser's own scaling.
WIDTH, HEIGHT = 900, 930
SS = 2

BG = (255, 255, 255)
INK = (24, 28, 34)
MUTED = (110, 117, 126)
FAINT = (174, 181, 190)
ARROW = (128, 134, 142)

# Blue marks the one metered step; green marks everything that is free and
# local. That distinction is the main thing a reader should take away.
PAID_FILL, PAID_LINE = (233, 240, 253), (48, 92, 176)
FREE_FILL, FREE_LINE = (233, 246, 237), (46, 125, 80)
PLAIN_FILL, PLAIN_LINE = (255, 255, 255), (150, 157, 165)
PANEL_FILL, PANEL_LINE = (250, 251, 252), (219, 224, 230)

# The loop edges are the point of the picture, so they get their own colour
# rather than sharing the grey used for everything entering and leaving it.
CYCLE = (208, 92, 42)


def _s(v):
    """Layout units -> device pixels."""
    return v * SS


def _st(xy):
    return tuple(v * SS for v in xy)


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
            return ImageFont.truetype(path, int(size * SS))
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
        (int(size * SS), int(size * SS)), Image.LANCZOS
    )


def _centre_dev(draw, cx, y, text, font, fill):
    """Centre already-scaled text at already-scaled coordinates."""
    b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (b[2] - b[0]) / 2, y), text, fill=fill, font=font)


def _text(draw, xy, text, fill=INK, size=16, bold=False):
    draw.text(_st(xy), text, fill=fill, font=_font(size, bold))


def _width(draw, text, size=16, bold=False):
    """Text width, back in layout units, for right-aligning."""
    return draw.textbbox((0, 0), text, font=_font(size, bold))[2] / SS


def _box(draw, xy, title, lines=(), fill=PLAIN_FILL, line=PLAIN_LINE,
         title_size=22, sub_size=15):
    x0, y0, x1, y1 = _st(xy)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=_s(11), fill=fill,
                           outline=line, width=_s(2))
    cx = (x0 + x1) / 2

    tf = _font(title_size, bold=True)
    sf = _font(sub_size)
    step = _s(sub_size + 6)
    title_h = draw.textbbox((0, 0), title, font=tf)[3]
    block = title_h + _s(9) + len(lines) * step
    y = (y0 + y1) / 2 - block / 2

    _centre_dev(draw, cx, y, title, tf, INK)
    y += title_h + _s(9)
    for text in lines:
        _centre_dev(draw, cx, y, text, sf, MUTED)
        y += step


def _arrow(draw, p0, p1, both=False, width=2, fill=None):
    fill = fill or ARROW
    p0, p1 = _st(p0), _st(p1)
    draw.line([p0, p1], fill=fill, width=_s(width))

    def head(tip, frm):
        dx, dy = tip[0] - frm[0], tip[1] - frm[1]
        n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / n, dy / n
        s = _s(10)
        draw.polygon([
            tip,
            (tip[0] - s * ux + s * 0.55 * -uy, tip[1] - s * uy + s * 0.55 * ux),
            (tip[0] - s * ux - s * 0.55 * -uy, tip[1] - s * uy - s * 0.55 * ux),
        ], fill=fill)

    head(p1, p0)
    if both:
        head(p0, p1)


def _elbow(draw, points, fill=None, width=2):
    """A polyline with a head on the final segment only."""
    fill = fill or ARROW
    draw.line([_st(p) for p in points], fill=fill, width=_s(width), joint="curve")
    _arrow(draw, points[-2], points[-1], fill=fill, width=width)


def _dashed_rect(draw, xy, colour=FAINT, dash=10, gap=8, width=2):
    x0, y0, x1, y1 = _st(xy)
    dash, gap, width = _s(dash), _s(gap), _s(width)
    for x in range(int(x0), int(x1), dash + gap):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=colour, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=colour, width=width)
    for y in range(int(y0), int(y1), dash + gap):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=colour, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=colour, width=width)


def _label(draw, cx, cy, text, colour=MUTED, size=15, bold=False):
    """A text label that knocks out whatever it sits on top of."""
    font = _font(size, bold=bold)
    cx, cy = _s(cx), _s(cy)
    b = draw.textbbox((0, 0), text, font=font)
    w, h = b[2] - b[0], b[3] - b[1]
    draw.rectangle((cx - w / 2 - _s(7), cy - h / 2 - _s(5),
                    cx + w / 2 + _s(7), cy + h / 2 + _s(7)), fill=BG)
    draw.text((cx - w / 2, cy - h / 2 - _s(1)), text, fill=colour, font=font)


def main() -> None:
    img = Image.new("RGB", (WIDTH * SS, HEIGHT * SS), BG)
    draw = ImageDraw.Draw(img)

    _text(draw, (36, 26), "MoviBot Architecture", INK, 32, bold=True)
    _text(draw, (36, 72),
          "One ReAct loop. The planner chooses the tools, their arguments, "
          "and when to stop \u2014 by not asking for one.", MUTED, 16)

    # ---- the loop ------------------------------------------------------
    # The frame carries the turn bound, because "bounded" is the honest
    # qualifier on "the model decides when to stop": it decides, up to here.
    frame = (210, 130, 680, 555)
    _dashed_rect(draw, frame)
    _label(draw, 445, 130,
           f"REACT LOOP  ·  the model repeats this until it stops, "
           f"or {loop.MAX_ROUNDS} times",
           colour=MUTED, size=15, bold=True)

    reason = (235, 175, 425, 300)
    act = (490, 175, 655, 300)
    observe = (490, 385, 655, 510)
    stop = (235, 385, 425, 510)

    _box(draw, reason, "Reason", ("Planner \u00b7 the only", "model call"),
         fill=PAID_FILL, line=PAID_LINE, title_size=27, sub_size=15)
    _box(draw, act, "Act", ("run the tools", "it asked for"),
         title_size=27, sub_size=15)
    _box(draw, observe, "Observe", ("results appended", "to the context"),
         title_size=27, sub_size=15)
    _box(draw, stop, "Stop?", ("did this turn ask", "for a tool?"),
         title_size=27, sub_size=15)
    # The exit condition is the model's own output, not a scripted check: a
    # turn that requests a tool continues the loop, and a turn that requests
    # none IS the answer. That is what "the model decides when to stop" means
    # mechanically, so the two edges have to say which way round it is.

    # Clockwise, with the return edge closing it. Without that edge the same
    # four boxes read as four stages.
    _arrow(draw, (427, 237), (486, 237), fill=CYCLE, width=3)
    _arrow(draw, (572, 302), (572, 381), fill=CYCLE, width=3)
    _arrow(draw, (488, 447), (429, 447), fill=CYCLE, width=3)
    _arrow(draw, (320, 383), (320, 304), fill=CYCLE, width=3)
    _label(draw, 390, 343, "Yes  →  loop again", colour=CYCLE, size=16, bold=True)

    # ---- request in, answer out ----------------------------------------
    request = (20, 187, 185, 287)
    answer = (705, 385, 878, 510)

    _box(draw, request, "User request",
         ("natural language,", "mixed constraints"), title_size=20, sub_size=14)
    _box(draw, answer, "Final answer",
         ("with the evidence", "it actually checked"), title_size=20, sub_size=14)

    _arrow(draw, (187, 237), (231, 237))

    # "Yes" leaves the loop from Stop?, under the frame, and comes up into the
    # answer -- so the one exit from the cycle is a decision, not a fall-through.
    # It runs right of x=390 so the tool connector, which drops on the left, has
    # nothing to cross.
    _elbow(draw, [(390, 512), (390, 590), (791, 590), (791, 514)],
           fill=CYCLE, width=3)
    _label(draw, 600, 590, "No  →  this turn is the answer",
           colour=CYCLE, size=16, bold=True)

    # ---- tools ---------------------------------------------------------
    # An inventory, not a chain: every tool is available on every turn. Left to
    # right is cheapest to dearest, which is the order the prompt recommends and
    # the planner usually adopts -- a preference, not a path, so nothing here
    # connects one tool to the next. The connector leaves the frame rather than
    # the Act box for the same reason: any turn may call any tool.
    panel = (36, 645, 864, 905)
    draw.rounded_rectangle(_st(panel), radius=_s(11), fill=PANEL_FILL,
                           outline=PANEL_LINE, width=_s(2))
    _text(draw, (60, 664), "TOOLS", MUTED, 15, bold=True)
    _text(draw, (60 + _width(draw, "TOOLS", 15, True) + 12, 665),
          "free · local · any order, any turn", FAINT, 14)

    # Left of x=390, so the "Yes" elbow above has nothing to knock out.
    _arrow(draw, (280, 557), (280, 641), both=True)
    _label(draw, 188, 599, "call · result", colour=MUTED, size=14)

    specs = [
        ("filter_catalog", "structured facts", "\U0001F5C2"),
        ("screen_out", "exhaustive scan", "\U0001F6AB"),
        ("search_plots", "vector search", "\U0001F50E"),
        ("read_synopses", "full plot text", "\U0001F4D6"),
    ]
    row_h, row_gap, row_y = 44, 7, 694
    for i, (key, sub, glyph) in enumerate(specs):
        y0 = row_y + i * (row_h + row_gap)
        draw.rounded_rectangle(_st((60, y0, 840, y0 + row_h)), radius=_s(8),
                               fill=FREE_FILL, outline=FREE_LINE, width=_s(1))
        tx = 76
        icon = _emoji(glyph, 20)
        if icon is not None:
            img.paste(icon, (_s(tx), _s(y0) + int((_s(row_h) - _s(20)) / 2)), icon)
            tx += 30
        _text(draw, (tx, y0 + 11), tools.TRACE_NAMES[key], INK, 19, bold=True)
        _text(draw, (824 - _width(draw, sub, 15), y0 + 15), sub, MUTED, 15)

    # Saved at full SS resolution rather than resampled back down. The browser
    # caps it at the container width either way, so the layout sizes above are
    # what the reader gets; keeping the extra pixels is what makes it sharp on
    # a retina display and legible when someone zooms in.
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}  ({WIDTH * SS}x{HEIGHT * SS}, "
          f"laid out as {WIDTH}x{HEIGHT} at {SS}x)")
    print(f"  loop bound: {loop.MAX_ROUNDS} model turns")
    print(f"  modules: {', '.join(tools.TRACE_NAMES[k] for k, _, _ in specs)}")


if __name__ == "__main__":
    main()
