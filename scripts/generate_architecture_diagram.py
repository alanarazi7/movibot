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
which is false -- LexicalScreen and PlotSearch read the same passage index --
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
WIDTH, HEIGHT = 918, 836
SS = 2

BG = (255, 255, 255)
INK = (24, 28, 34)
MUTED = (110, 117, 126)
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
        (MAYBE_FILL, MAYBE_LINE, "embedding call \u00b7 ~$0.0000002 each"),
        (FREE_FILL, FREE_LINE, "pure Python \u00b7 free"),
        (IO_FILL, IO_LINE, "local data \u00b7 read from disk once"),
    ]
    lw = max(_width(draw, label, 14) for _, _, label in entries)
    lx = WIDTH - 36 - lw - 20
    for i, (fill, line, label) in enumerate(entries):
        ly = 30 + i * 22
        _swatch(draw, lx, ly, fill, line)
        _text(draw, (lx + 20, ly - 2), label, MUTED, 14)

    # ---- the pipeline --------------------------------------------------
    # A fixed route, drawn as one. Which stages run depends only on which
    # fields the plan filled in, so two runs of the same request produce the
    # same shape of trace; what varies is what the films turn out to say.
    request = (30, 196, 190, 286)
    decomp  = (222, 176, 410, 306)
    stages  = (445, 150, 690, 332)
    walk    = (222, 396, 410, 526)
    answer  = (445, 396, 690, 526)
    reply   = (725, 396, 882, 526)

    _box(draw, request, "Request", ("natural language,", "mixed constraints"),
         title_size=21, sub_size=14)

    _box(draw, decomp, "QueryDecomposer", ("reads it once,", "returns a plan"),
         fill=PAID_FILL, line=PAID_LINE, title_size=19, sub_size=14)

    # The three stages share a frame because they are one pass, not three
    # decisions: the plan says which of them have anything to do.
    _dashed_rect(draw, stages)
    _text(draw, (462, 164), "PLAN DRIVES THESE", MUTED, 13, bold=True)
    for k, (name, sub, fill, line) in enumerate([
        ("CatalogFilter", "columns \u00b7 exact", FREE_FILL, FREE_LINE),
        ("LexicalScreen", "word scan \u00b7 every plot", FREE_FILL, FREE_LINE),
        ("ShortlistFusion", "one embedding per condition", MAYBE_FILL, MAYBE_LINE),
    ]):
        y0 = 188 + k * 46
        draw.rounded_rectangle(_st((462, y0, 673, y0 + 38)), radius=_s(7),
                               fill=fill, outline=line, width=_s(1))
        _text(draw, (474, y0 + 4), name, INK, 16, bold=True)
        _text(draw, (474, y0 + 21), sub, MUTED, 12)

    _box(draw, walk, "CandidateWalk", ("best first, until 3 pass", "or 10 are read"),
         fill=FREE_FILL, line=FREE_LINE, title_size=21, sub_size=14)
    _box(draw, answer, "Answerer", ("writes the reply from", "what was accepted"),
         fill=PAID_FILL, line=PAID_LINE, title_size=21, sub_size=14)
    _box(draw, reply, "Reply", ("with the evidence", "it checked"),
         title_size=20, sub_size=13)

    # The Verifier hangs off the walk: one call per film, which is where a
    # verifying request spends nearly all of its money.
    verifier = (222, 578, 690, 646)
    draw.rounded_rectangle(_st(verifier), radius=_s(9),
                           fill=PAID_FILL, outline=PAID_LINE, width=_s(2))
    _text(draw, (244, 592), "Verifier", INK, 21, bold=True)
    _text(draw, (244 + _width(draw, "Verifier", 21, True) + 16, 598),
          "one film \u00b7 every condition \u00b7 quotes what decides it",
          MUTED, 14)

    _arrow(draw, (192, 241), (218, 241))
    _arrow(draw, (412, 241), (441, 241), fill=CYCLE, width=3)
    # down the right of the stage frame and back left into the walk
    _elbow(draw, [(567, 334), (567, 360), (316, 360), (316, 392)],
           fill=CYCLE, width=3)
    _arrow(draw, (316, 528), (316, 574), fill=CYCLE, width=3)
    _elbow(draw, [(690, 612), (706, 612), (706, 500), (688, 500)],
           fill=CYCLE, width=3)
    _label(draw, 400, 366, "candidates", colour=MUTED, size=13)
    _label(draw, 250, 552, "one call per film", colour=MUTED, size=13)
    _label(draw, 620, 552, "verdicts", colour=MUTED, size=13)
    _arrow(draw, (692, 461), (721, 461), fill=CYCLE, width=3)

    # The check sits on the last edge because it is the last thing that
    # happens, and it is the only thing that can stop a reply.
    _label(draw, 694, 434, "answer check", colour=FREE_LINE, size=12, bold=True)

    # ---- what the stages read -----------------------------------------
    # The data layer, which every earlier picture left out. It is not a step
    # and nothing calls it; it is what the stages are reading when they look
    # free, and the reason they are.
    panel = (36, 690, 882, 806)
    draw.rounded_rectangle(_st(panel), radius=_s(11), fill=PANEL_FILL,
                           outline=PANEL_LINE, width=_s(2))
    _text(draw, (60, 709), "LOCAL DATA", MUTED, 15, bold=True)
    _text(draw, (60 + _width(draw, "LOCAL DATA", 15, True) + 12, 710),
          "committed to the repo \u00b7 loaded once \u00b7 nothing fetched at request time",
          FAINT, 14)

    for k, (name, sub) in enumerate([
        ("Catalog", f"{len(catalog.movies())} films \u00b7 columns, ratings, runtimes"),
        ("Passage index", f"{store.coverage()['chunks']:,} vectors of plot text "
                          f"\u00b7 {store.coverage()['dim']}-dim"),
    ]):
        x0 = 60 + k * 420
        draw.rounded_rectangle(_st((x0, 736, x0 + 396, 786)), radius=_s(8),
                               fill=IO_FILL, outline=IO_LINE, width=_s(1))
        _text(draw, (x0 + 16, 746), name, INK, 17, bold=True)
        _text(draw, (x0 + 16, 766), sub, MUTED, 13)

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
    print(f"  roles: QueryDecomposer, Verifier, Answerer")
    print(f"  stages: {', '.join(sorted(tools.TRACE_NAMES.values()))}")


if __name__ == "__main__":
    main()
