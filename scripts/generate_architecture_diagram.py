"""Renders assets/architecture.png from MoviBot's actual modules and data.

Offline only (PIL, no network or model calls).

Two things this diagram must not drift from, so both are read rather than typed:
module names come from agent.tools.TRACE_NAMES and the turn bound from
agent.loop.MAX_ROUNDS. The assignment requires identical naming across the
diagram, the /api/execute steps trace, and agent_info.json, so none of it is
typed here.

WHY THE CYCLE IS THE SUBJECT

The previous version drew the planner fanning out to four tools. Everything in
it was true, but with no return edge and no turn counter it read as one-shot
dispatch -- a router in front of a fixed pipeline. That is the wrong takeaway:
agent/loop.py offers every tool on every turn, the model picks which and how
many, and the loop exits only when it emits no tool call. So the cycle, not the
fan-out, is what the diagram has to show: Reason -> Act -> Observe -> Stop?,
with the "No" edge closing it and the turn bound printed on the frame.

The tools are drawn as an inventory the Act step reaches into, deliberately
*not* as a left-to-right chain. There is no code path that runs them in order;
the ordering is a preference stated in the system prompt, and a diagram that
chains them asserts a mechanism that does not exist.

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

WIDTH, HEIGHT = 1240, 560

BG = (255, 255, 255)
INK = (24, 28, 34)
MUTED = (118, 124, 132)
FAINT = (176, 183, 191)
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
         title_size=22, sub_size=15):
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=line, width=2)
    x0, y0, x1, y1 = xy
    cx = (x0 + x1) / 2

    tf = _font(title_size, bold=True)
    sf = _font(sub_size)
    step = sub_size + 5
    block = (draw.textbbox((0, 0), title, font=tf)[3] + 6) + len(lines) * step
    y = (y0 + y1) / 2 - block / 2

    _centre(draw, cx, y, title, tf, INK)
    y += draw.textbbox((0, 0), title, font=tf)[3] + 8
    for text in lines:
        _centre(draw, cx, y, text, sf, MUTED)
        y += step


def _arrow(draw, p0, p1, both=False, width=2, fill=None):
    fill = fill or ARROW
    draw.line([p0, p1], fill=fill, width=width)

    def head(tip, frm):
        dx, dy = tip[0] - frm[0], tip[1] - frm[1]
        n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / n, dy / n
        s = 9
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
    draw.line(points, fill=fill, width=width, joint="curve")
    _arrow(draw, points[-2], points[-1], fill=fill, width=width)


def _dashed_rect(draw, xy, colour=FAINT, dash=9, gap=7, width=2):
    x0, y0, x1, y1 = xy
    for x in range(int(x0), int(x1), dash + gap):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=colour, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=colour, width=width)
    for y in range(int(y0), int(y1), dash + gap):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=colour, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=colour, width=width)


def _label(draw, cx, cy, text, colour=MUTED, size=15, bold=False):
    font = _font(size, bold=bold)
    b = draw.textbbox((0, 0), text, font=font)
    w, h = b[2] - b[0], b[3] - b[1]
    draw.rectangle((cx - w / 2 - 6, cy - h / 2 - 4, cx + w / 2 + 6, cy + h / 2 + 6),
                   fill=BG)
    draw.text((cx - w / 2, cy - h / 2 - 1), text, fill=colour, font=font)


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.text((40, 24), "MoviBot Architecture", fill=INK, font=_font(32, bold=True))
    draw.text((40, 64), "One ReAct loop. The planner chooses the tools, their "
                        "arguments, and when to stop.",
              fill=MUTED, font=_font(16))

    # ---- the loop ------------------------------------------------------
    # The frame carries the turn bound, because "bounded" is the honest
    # qualifier on "the model decides when to stop": it decides, up to here.
    frame = (300, 128, 800, 452)
    _dashed_rect(draw, frame)
    _label(draw, 550, 128,
           f"REACT LOOP  ·  at most {loop.MAX_ROUNDS} model turns",
           colour=MUTED, size=14, bold=True)

    reason = (330, 168, 520, 268)
    act = (600, 168, 770, 268)
    observe = (600, 322, 770, 422)
    stop = (330, 322, 520, 422)

    _box(draw, reason, "Reason",
         ("Planner", "the only metered step"),
         fill=PAID_FILL, line=PAID_LINE, title_size=23, sub_size=14)
    _box(draw, act, "Act", ("call one or", "more tools"),
         title_size=23, sub_size=14)
    _box(draw, observe, "Observe", ("read results,", "update working set"),
         title_size=23, sub_size=14)
    _box(draw, stop, "Stop?", ("enough evidence", "to answer?"),
         title_size=23, sub_size=14)

    # Clockwise, with the return edge closing it. Without that edge the same
    # four boxes read as four stages.
    _arrow(draw, (522, 218), (596, 218), fill=CYCLE, width=3)
    _arrow(draw, (685, 270), (685, 318), fill=CYCLE, width=3)
    _arrow(draw, (598, 372), (524, 372), fill=CYCLE, width=3)
    _arrow(draw, (392, 320), (392, 272), fill=CYCLE, width=3)
    _label(draw, 462, 296, "No  →  loop", colour=CYCLE, size=14, bold=True)

    # ---- request in, answer out ----------------------------------------
    request = (36, 178, 258, 258)
    answer = (952, 400, 1200, 496)

    _box(draw, request, "User request", ("natural language,", "mixed constraints"),
         title_size=20, sub_size=14)
    _box(draw, answer, "Final answer", ("with the evidence", "it actually checked"),
         title_size=20, sub_size=14)

    _arrow(draw, (260, 218), (326, 218))

    # "Yes" leaves the loop from Stop?, under the frame, and comes up into the
    # answer -- so the one exit from the cycle is a decision, not a fall-through.
    _elbow(draw, [(392, 424), (392, 520), (1076, 520), (1076, 500)],
           fill=CYCLE, width=3)
    _label(draw, 700, 520, "Yes  →  answer", colour=CYCLE, size=14, bold=True)

    # ---- tools ---------------------------------------------------------
    # An inventory, not a chain: every tool is available on every turn. Left to
    # right is cheapest to dearest, which is the order the prompt recommends and
    # the planner usually adopts -- a preference, not a path, so nothing here
    # connects one tool to the next.
    panel = (852, 128, 1200, 344)
    draw.rounded_rectangle(panel, radius=10, fill=PANEL_FILL, outline=PANEL_LINE,
                           width=2)
    draw.text((874, 146), "TOOLS", fill=MUTED, font=_font(14, bold=True))
    draw.text((874 + draw.textbbox((0, 0), "TOOLS", font=_font(14, bold=True))[2] + 10,
               146), "free · local · any order, any turn",
              fill=FAINT, font=_font(13))

    specs = [
        ("filter_catalog", "structured facts", "\U0001F5C2"),
        ("screen_out", "exhaustive scan", "\U0001F6AB"),
        ("search_plots", "vector search", "\U0001F50E"),
        ("read_synopses", "full plot text", "\U0001F4D6"),
    ]
    row_h, row_y = 40, 176
    for i, (key, sub, glyph) in enumerate(specs):
        y0 = row_y + i * (row_h + 8)
        draw.rounded_rectangle((874, y0, 1178, y0 + row_h), radius=7,
                               fill=FREE_FILL, outline=FREE_LINE, width=1)
        tx = 890
        icon = _emoji(glyph, 17)
        if icon is not None:
            img.paste(icon, (tx, y0 + int((row_h - 17) / 2)), icon)
            tx += 25
        nf = _font(16, bold=True)
        draw.text((tx, y0 + 11), tools.TRACE_NAMES[key], fill=INK, font=nf)
        sf = _font(13)
        sw = draw.textbbox((0, 0), sub, font=sf)[2]
        draw.text((1166 - sw, y0 + 14), sub, fill=MUTED, font=sf)

    # One double-headed edge: Act calls a tool, the result comes back for the
    # planner to observe. Drawn as a pair rather than separate call/return lines,
    # which turned into a thicket of overlapping dashes in an earlier version.
    _arrow(draw, (772, 210), (848, 210), both=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  loop bound: {loop.MAX_ROUNDS} model turns")
    print(f"  modules: {', '.join(tools.TRACE_NAMES[k] for k, _, _ in specs)}")


if __name__ == "__main__":
    main()
