"""Renders assets/architecture.png from MoviBot's actual modules and data.

Offline only (PIL, no network or model calls).

Two things this diagram must not drift from, so both are read rather than typed:
module names come from agent.tools.TRACE_NAMES and the turn bound from
agent.loop.MAX_ROUNDS -- the assignment requires identical naming across the
diagram, the /api/execute steps trace, and agent_info.json -- and the row counts
come from data_ready/, so a re-prepared catalog cannot leave a stale figure on
the page.

Usage: python scripts/generate_architecture_diagram.py
"""

import os
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import loop, tools  # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA = os.path.join(_ROOT, "data_preprocessing", "data_ready")
OUTPUT_PATH = os.path.join(_ROOT, "assets", "architecture.png")

WIDTH, HEIGHT = 1280, 720

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


def _centre(draw, cx, y, text, font, fill):
    b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (b[2] - b[0]) / 2, y), text, fill=fill, font=font)


def _box(draw, xy, title, lines=(), fill=PLAIN_FILL, line=PLAIN_LINE,
         title_size=17):
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=line, width=2)
    x0, y0, x1, y1 = xy
    cx = (x0 + x1) / 2

    tf = _font(title_size, bold=True)
    sf = _font(12.5)
    block = (draw.textbbox((0, 0), title, font=tf)[3] + 4) + len(lines) * 16
    y = (y0 + y1) / 2 - block / 2

    _centre(draw, cx, y, title, tf, INK)
    y += draw.textbbox((0, 0), title, font=tf)[3] + 6
    for text in lines:
        _centre(draw, cx, y, text, sf, MUTED)
        y += 16


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
    font = font or _font(12.5)
    b = draw.textbbox((0, 0), text, font=font)
    w, h = b[2] - b[0], b[3] - b[1]
    draw.rectangle((cx - w / 2 - 5, cy - h / 2 - 3, cx + w / 2 + 5, cy + h / 2 + 5), fill=BG)
    draw.text((cx - w / 2, cy - h / 2 - 1), text, fill=MUTED, font=font)


def _band(draw, y, text, note):
    f, fn = _font(11, bold=True), _font(11)
    draw.text((40, y), text.upper(), fill=MUTED, font=f)
    w = draw.textbbox((0, 0), text.upper(), font=f)[2]
    draw.text((40 + w + 12, y), note, fill=(180, 186, 193), font=fn)


def counts() -> dict:
    """Figures for the data layer, read from the prepared files."""
    catalog = pd.read_csv(os.path.join(_DATA, "supabase_movies.csv"))
    synopses = pd.read_csv(os.path.join(_DATA, "pinecone_candidates.csv"))
    wiki = pd.read_csv(os.path.join(_DATA, "wikipedia_cache.csv"))
    chunks = pd.read_parquet(os.path.join(_DATA, "plot_chunks.parquet"))

    mpst = set(synopses["movie_id"])
    wiki_plot = set(wiki.loc[wiki["plot_text"].notna(), "id"])

    return {
        "films": len(catalog),
        "columns": catalog.shape[1],
        "passages": len(chunks),
        "indexed_films": chunks["movie_id"].nunique(),
        "readable": len(mpst | wiki_plot),
    }


def main() -> None:
    n = counts()
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.text((40, 30), "MoviBot Architecture", fill=INK, font=_font(27, bold=True))
    draw.text(
        (40, 68),
        f"A planner with three tools over {n['films']} Disney and Pixar feature films. "
        f"Only the planner costs anything, and it is capped at {loop.MAX_ROUNDS} turns "
        f"per request.",
        fill=MUTED, font=_font(14),
    )

    # ---- request / planner / answer -----------------------------------
    req = (40, 118, 250, 178)
    plan = (470, 108, 810, 190)
    ans = (1070, 118, 1240, 178)

    _box(draw, req, "User request")
    _box(draw, plan, tools.TRACE_NAMES.get("planner", "Planner"),
         ("decides which tools the question needs,", "and when it has enough to answer"),
         fill=PAID_FILL, line=PAID_LINE, title_size=18)
    _box(draw, ans, "Final answer")

    _arrow(draw, (250, 148), (466, 148))
    _arrow(draw, (814, 148), (1066, 148))
    _label(draw, 940, 148, "no more tools needed")

    draw.text((476, 194), f"metered · at most {loop.MAX_ROUNDS} model turns",
              fill=PAID_LINE, font=_font(11.5))

    # ---- tools ---------------------------------------------------------
    _band(draw, 232, "Tools", "called by the planner, run locally, no model cost")

    specs = [
        ("filter_catalog", ("answers from columns",
                            "year · genre · studio · language")),
        ("search_plots", ("answers from meaning",
                          "passage-level semantic search")),
        ("read_synopses", ("answers from full text",
                           "who dies, who betrays whom")),
    ]
    tw, gap = 356, 42
    start = (WIDTH - (tw * 3 + gap * 2)) / 2
    ty0, ty1 = 268, 348

    tool_boxes = []
    for i, (key, sub) in enumerate(specs):
        x0 = start + i * (tw + gap)
        box = (x0, ty0, x0 + tw, ty1)
        tool_boxes.append(box)
        _box(draw, box, tools.TRACE_NAMES[key], sub, fill=FREE_FILL, line=FREE_LINE)

    # One bus off the planner, then a two-way link into each tool: the planner
    # calls, the result comes back to the planner. Drawn as double-headed
    # arrows rather than separate call/return lines, which turned into a
    # thicket of overlapping dashes in the previous version.
    bus_y = 246
    draw.line([(640, 190), (640, bus_y)], fill=ARROW, width=2)
    draw.line([((tool_boxes[0][0] + tool_boxes[0][2]) / 2, bus_y),
               ((tool_boxes[-1][0] + tool_boxes[-1][2]) / 2, bus_y)], fill=ARROW, width=2)
    for box in tool_boxes:
        cx = (box[0] + box[2]) / 2
        _arrow(draw, (cx, bus_y), (cx, ty0 - 2), both=True)

    # ---- data ----------------------------------------------------------
    _band(draw, 392, "Data", "prepared offline")

    data = [
        ("Catalog", (f"{n['films']} films × {n['columns']} columns",
                     "CSV locally · Supabase in cloud mode")),
        ("Passage index", (f"{n['passages']:,} passages from {n['indexed_films']} films",
                           "1536-d, committed matrix or Pinecone")),
        ("Plot texts", (f"{n['readable']} films readable in full",
                        "MPST synopsis, else Wikipedia plot")),
    ]
    dy0, dy1 = 428, 508
    for box, (title, sub) in zip(tool_boxes, data):
        dbox = (box[0], dy0, box[2], dy1)
        _box(draw, dbox, title, sub, fill=DATA_FILL, line=DATA_LINE, title_size=15)
        cx = (box[0] + box[2]) / 2
        _arrow(draw, (cx, ty1 + 2), (cx, dy0 - 2))
        _label(draw, cx, ty1 + 20, "reads")

    # ---- guardrails ----------------------------------------------------
    gy = 552
    draw.rounded_rectangle((40, gy, WIDTH - 40, gy + 108), radius=10,
                           fill=(252, 250, 244), outline=(228, 218, 196), width=1)
    draw.text((60, gy + 16), "Guardrails live in the data and the tool code, never in the prompt",
              fill=(122, 96, 40), font=_font(14, bold=True))
    draw.text(
        (60, gy + 42),
        "The model cannot forget them and a bad plan cannot route around them:\n"
        "  ·  the catalog holds feature films only — shorts under 45 minutes were dropped at preparation time\n"
        "  ·  results are ordered by a vote-weighted rating, never the raw average, so a 5-vote film cannot top a list",
        fill=(140, 116, 62), font=_font(12.5), spacing=5,
    )

    draw.text(
        (40, HEIGHT - 34),
        "All data is committed to the repo and read from disk — no API call is made at query time.  ·  "
        "Every box above appears by this exact name in the `module` field of the /api/execute steps trace.",
        fill=(178, 184, 191), font=_font(12),
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  modules: {', '.join(tools.TRACE_NAMES[k] for k, _ in specs)}")
    print(f"  data:    {n}")


if __name__ == "__main__":
    main()
