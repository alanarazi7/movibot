"""Renders assets/architecture.png from MoviBot's actual module list.

Offline only (PIL, no network or model calls). The labels here must stay in
sync with the `module` values logged by agent/loop.py -- which come from
agent.tools.TRACE_NAMES -- and with agent_info.json, since the assignment
requires identical naming across the diagram, the /api/execute steps trace,
and the agent description.

Usage: python scripts/generate_architecture_diagram.py
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import loop, tools  # noqa: E402

WIDTH, HEIGHT = 1240, 760
BG = (255, 255, 255)
BOX_FILL = (235, 242, 255)
BOX_OUTLINE = (60, 90, 160)
FREE_FILL = (232, 246, 236)
FREE_OUTLINE = (52, 130, 82)
TEXT_COLOR = (20, 20, 20)
MUTED = (110, 110, 110)
ARROW_COLOR = (90, 90, 90)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "architecture.png")


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


def _box(draw, xy, title, subtitle=None, free=False):
    fill = FREE_FILL if free else BOX_FILL
    outline = FREE_OUTLINE if free else BOX_OUTLINE
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=outline, width=2)

    x0, y0, x1, y1 = xy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tf, sf = _font(17, bold=True), _font(13)

    if subtitle:
        b = draw.textbbox((0, 0), title, font=tf)
        draw.text((cx - (b[2] - b[0]) / 2, cy - 18), title, fill=TEXT_COLOR, font=tf)
        for i, line in enumerate(subtitle.split("\n")):
            b = draw.textbbox((0, 0), line, font=sf)
            draw.text((cx - (b[2] - b[0]) / 2, cy + 2 + i * 16), line, fill=MUTED, font=sf)
    else:
        b = draw.textbbox((0, 0), title, font=tf)
        draw.text((cx - (b[2] - b[0]) / 2, cy - (b[3] - b[1]) / 2), title,
                  fill=TEXT_COLOR, font=tf)


def _arrow(draw, p0, p1, label=None, dashed=False):
    if dashed:
        x0, y0 = p0
        x1, y1 = p1
        total = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-6)
        steps = int(total // 10)
        for i in range(steps):
            if i % 2:
                continue
            a = (x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
            b = (x0 + (x1 - x0) * (i + 1) / steps, y0 + (y1 - y0) * (i + 1) / steps)
            draw.line([a, b], fill=ARROW_COLOR, width=2)
    else:
        draw.line([p0, p1], fill=ARROW_COLOR, width=2)

    x1, y1 = p1
    dx, dy = x1 - p0[0], y1 - p0[1]
    length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    s = 8
    left = (x1 - s * ux + s * 0.5 * -uy, y1 - s * uy + s * 0.5 * ux)
    right = (x1 - s * ux - s * 0.5 * -uy, y1 - s * uy - s * 0.5 * ux)
    draw.polygon([p1, left, right], fill=ARROW_COLOR)

    if label:
        f = _font(13)
        mx, my = (p0[0] + x1) / 2, (p0[1] + y1) / 2
        b = draw.textbbox((0, 0), label, font=f)
        draw.rectangle(
            (mx - (b[2] - b[0]) / 2 - 4, my - 9, mx + (b[2] - b[0]) / 2 + 4, my + 9),
            fill=BG,
        )
        draw.text((mx - (b[2] - b[0]) / 2, my - 8), label, fill=ARROW_COLOR, font=f)


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.text((40, 26), "MoviBot Architecture", fill=TEXT_COLOR, font=_font(28, bold=True))
    draw.text(
        (40, 64),
        f"Tool-calling agent over a 238-film catalog. Bounded at "
        f"{loop.MAX_ROUNDS} model turns per request; all tools run locally at no cost.",
        fill=MUTED, font=_font(14),
    )

    user = (40, 130, 250, 190)
    planner = (500, 130, 740, 200)
    answer = (990, 130, 1200, 190)

    _box(draw, user, "User Request")
    _box(draw, planner, "Planner", "LLM chooses tools\nand when to stop")
    _box(draw, answer, "Final Answer")

    _arrow(draw, (250, 160), (500, 160))
    _arrow(draw, (740, 160), (990, 160), label="no more tools")

    # Tools row, labelled with the same names logged in the steps trace.
    specs = [
        (tools.TRACE_NAMES["filter_catalog"], "structured columns\nyear/genre/language"),
        (tools.TRACE_NAMES["search_plots"], "semantic search\nlocal E5 vectors"),
        (tools.TRACE_NAMES["read_synopses"], "full plot text\nwhat happens in it"),
    ]
    y0, y1 = 400, 480
    w, gap = 300, 60
    start = (WIDTH - (w * len(specs) + gap * (len(specs) - 1))) / 2

    boxes = []
    for i, (name, sub) in enumerate(specs):
        x0 = start + i * (w + gap)
        box = (x0, y0, x0 + w, y1)
        boxes.append(box)
        _box(draw, box, name, sub, free=True)

    for box in boxes:
        cx = (box[0] + box[2]) / 2
        _arrow(draw, (cx, y0 - 100), (cx, y0))
        _arrow(draw, (cx + 14, y0), (cx + 14, y0 - 100), dashed=True)

    draw.text((start - 4, y0 - 130), "call", fill=ARROW_COLOR, font=_font(13))
    draw.text((boxes[-1][2] - 60, y0 - 130), "results", fill=ARROW_COLOR, font=_font(13))
    draw.line([(620, 200), (620, 300)], fill=ARROW_COLOR, width=2)
    draw.line([(240, 300), (1000, 300)], fill=ARROW_COLOR, width=2)

    draw.text(
        (40, 540),
        "The Planner picks only the tools a query needs, so simple requests finish in one round\n"
        "and hard ones in three. Two guardrails live below this diagram, in the data and the tool\n"
        "code rather than in the prompt, so no plan can bypass them: the catalog holds feature\n"
        "films only (shorts under 45 minutes are dropped at preparation time), and results are\n"
        "ordered by a vote-count-weighted rating rather than the raw average.\n\n"
        "Every box above appears by this exact name in the /api/execute steps trace.",
        fill=(80, 80, 80), font=_font(14), spacing=5,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}  ({', '.join(n for n, _ in specs)})")


if __name__ == "__main__":
    main()
