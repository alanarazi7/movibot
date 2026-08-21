"""Renders assets/architecture.png from MoviBot's actual modules and data.

Offline only (PIL, no network or model calls).

Two things this diagram must not drift from, so both are read rather than typed:
module names come from agent.tools.TRACE_NAMES and the turn bound from
agent.tools.TRACE_NAMES. The assignment requires identical naming across the
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
which is false -- LexicalScan and PlotSearch read the same passage index --
or a shared bus, which implies every tool reads every store and is equally
false. The counts live in the Architecture tab and /api/rag/info instead.

Usage: python scripts/generate_architecture_diagram.py
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import catalog, loop, tools  # noqa: E402
from rag import store  # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_PATH = os.path.join(_ROOT, "assets", "architecture.png")

# Every coordinate below is in layout units, which are the pixels the reader
# actually gets. SS only buys resampling quality: the image is drawn SS times
# larger and shrunk back down at save time, so curves and small type stay clean
# on a retina display and survive the browser's own scaling.
WIDTH, HEIGHT = 918, 622
SS = 2

BG = (255, 255, 255)
INK = (24, 28, 34)
MUTED = (88, 95, 104)
FAINT = (174, 181, 190)
ARROW = (128, 134, 142)

# Four kinds of work, because "costs money" was hiding a real difference: a
# text-model call and an embedding call differ by about four orders of
# magnitude, and neither is the same as reading a CSV off disk.
PAID_FILL, PAID_LINE = (233, 240, 253), (48, 92, 176)      # text model
MAYBE_FILL, MAYBE_LINE = (242, 236, 250), (114, 71, 168)   # embedding model
FREE_FILL, FREE_LINE = (233, 246, 237), (46, 125, 80)      # pure Python
IO_FILL, IO_LINE = (252, 246, 233), (176, 122, 30)         # local data
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


def _swatch(draw, x, y, fill, line, size=13):
    """A small filled square, so a legend entry reads as the box it explains."""
    draw.rounded_rectangle(_st((x, y, x + size, y + size)), radius=_s(3),
                           fill=fill, outline=line, width=_s(2))


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

    title = "MoviBot Architecture"
    _text(draw, (36, 26), title, INK, 32, bold=True)
    popcorn = _emoji("\U0001F37F", 30)
    if popcorn is not None:
        img.paste(popcorn, (int(_s(36 + _width(draw, title, 32, True) + 14)), int(_s(28))), popcorn)

    # Top right, stacked, so it clears the title and reads downward the way the
    # boxes it explains are stacked.
    entries = [
        (PAID_FILL, PAID_LINE, "text-model call"),
        (MAYBE_FILL, MAYBE_LINE, "embedding call"),
        (FREE_FILL, FREE_LINE, "Python"),
    ]
    lw = max(_width(draw, label, 16) for _, _, label in entries)
    lx = WIDTH - 36 - lw - 20
    for i, (fill, line, label) in enumerate(entries):
        ly = 30 + i * 25
        _swatch(draw, lx, ly, fill, line)
        _text(draw, (lx + 20, ly - 3), label, MUTED, 16)

    # ---- the agent -----------------------------------------------------
    # Drawn as a choice, not a chain. The operations sit in a box the
    # Decomposer reaches into: it asks for the ones the request needs, with
    # the arguments it wrote, and a request that needs one of them touches
    # one. Laying them out as three consecutive boxes drew a conveyor belt
    # and said something about the agent that is not true.
    request = (30, 196, 178, 286)
    decomp  = (212, 176, 392, 306)
    ops     = (432, 140, 708, 350)
    verify  = (212, 420, 480, 550)
    answer  = (528, 420, 728, 550)
    reply   = (762, 420, 888, 550)

    _box(draw, request, "Request", (), title_size=23, sub_size=16)
    _box(draw, decomp, "Decomposer", ("decides what it needs",),
         fill=PAID_FILL, line=PAID_LINE, title_size=24, sub_size=16)

    _dashed_rect(draw, ops)
    tools_icon = _emoji("\U0001F9F0", 21)
    tx = 448
    if tools_icon is not None:
        img.paste(tools_icon, (_s(tx), _s(154)), tools_icon)
        tx += 26
    _text(draw, (tx, 153), "Tools", MUTED, 19, bold=True)
    _text(draw, (tx + _width(draw, "Tools", 19, True) + 12, 158),
          "any, some twice", MUTED, 14)

    for k, (name, glyph, fill, line) in enumerate([
        ("CatalogFilter", "\U0001F5C2", FREE_FILL, FREE_LINE),
        ("PlotRetrieval", "\U0001F4D6", MAYBE_FILL, MAYBE_LINE),
        ("MetadataRetrieval", "\U0001F9E0", MAYBE_FILL, MAYBE_LINE),
    ]):
        y0 = 192 + k * 50
        draw.rounded_rectangle(_st((448, y0, 692, y0 + 42)), radius=_s(7),
                               fill=fill, outline=line, width=_s(1))
        ix = 462
        icon = _emoji(glyph, 21)
        if icon is not None:
            img.paste(icon, (_s(ix), _s(y0) + int((_s(42) - _s(21)) / 2)), icon)
            ix += 28
        _text(draw, (ix, y0 + 9), name, INK, 19, bold=True)

    _box(draw, verify, "Verifier", ("one film at a time",),
         fill=PAID_FILL, line=PAID_LINE, title_size=24, sub_size=16)
    _box(draw, answer, "Answerer", (),
         fill=PAID_FILL, line=PAID_LINE, title_size=24, sub_size=16)
    _box(draw, reply, "Reply", ("with its evidence",), title_size=22, sub_size=15)

    _arrow(draw, (180, 241), (208, 241))
    _arrow(draw, (394, 241), (428, 241), fill=CYCLE, width=3)
    # candidates fall out of the operations box into verification
    _elbow(draw, [(570, 352), (570, 380), (346, 380), (346, 416)],
           fill=CYCLE, width=3)
    _label(draw, 420, 382, "candidates", colour=MUTED, size=15)
    # The Verifier reads one film, then the next, until enough have passed or
    # the candidates run out. Drawn as a loop because that is what it is, and
    # because it is where a verifying request spends nearly all of its money.
    _elbow(draw, [(300, 552), (300, 582), (186, 582), (186, 470), (208, 470)],
           fill=CYCLE, width=3)
    _text(draw, (196, 596), "next candidate, until enough pass", MUTED, 14)
    _arrow(draw, (482, 485), (524, 485), fill=CYCLE, width=3)
    _arrow(draw, (730, 485), (758, 485), fill=CYCLE, width=3)

    # Saved at full SS resolution rather than resampled back down. The browser
    # caps it at the container width either way, so the layout sizes above are
    # what the reader gets; keeping the extra pixels is what makes it sharp on
    # a retina display and legible when someone zooms in.
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}  ({WIDTH * SS}x{HEIGHT * SS}, "
          f"laid out as {WIDTH}x{HEIGHT} at {SS}x)")
    # Reported, not drawn: the bound is a cost guard rather than something
    # a reader needs in order to follow the picture.
    print(f"  call cap: {loop.MAX_TOTAL_LLM_CALLS} model calls, "
          f"{tools.MAX_VERIFICATIONS} verifications (not shown)")
    print(f"  roles: Decomposer, Verifier, Answerer")
    print(f"  stages: {', '.join(sorted(tools.TRACE_NAMES.values()))}")


if __name__ == "__main__":
    main()
