"""Renders assets/architecture.png from the fixed MoviBot module list.

Offline only (PIL, no network/model calls). Module names here must stay in
sync with agent/react_loop.py's logged `module` values and agent_info.json,
since the assignment requires identical naming across the diagram, the
/api/execute steps trace, and any descriptions.

Usage: python scripts/generate_architecture_diagram.py
"""

import os

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 800
BG = (255, 255, 255)
BOX_FILL = (235, 242, 255)
BOX_OUTLINE = (60, 90, 160)
TEXT_COLOR = (20, 20, 20)
ARROW_COLOR = (90, 90, 90)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "architecture.png")


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _box(draw: ImageDraw.ImageDraw, xy, label: str, font: ImageFont.FreeTypeFont) -> None:
    draw.rounded_rectangle(xy, radius=12, fill=BOX_FILL, outline=BOX_OUTLINE, width=2)
    x0, y0, x1, y1 = xy
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 - th / 2), label, fill=TEXT_COLOR, font=font)


def _arrow(draw: ImageDraw.ImageDraw, p0, p1) -> None:
    draw.line([p0, p1], fill=ARROW_COLOR, width=2)
    x1, y1 = p1
    dx, dy = x1 - p0[0], y1 - p0[1]
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    size = 8
    left = (x1 - size * ux + size * 0.5 * -uy, y1 - size * uy + size * 0.5 * ux)
    right = (x1 - size * ux - size * 0.5 * -uy, y1 - size * uy - size * 0.5 * ux)
    draw.polygon([p1, left, right], fill=ARROW_COLOR)


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    title_font = _font(28)
    box_font = _font(18)
    small_font = _font(14)

    draw.text((40, 20), "MoviBot Architecture — ReAct Loop", fill=TEXT_COLOR, font=title_font)
    draw.text(
        (40, 58),
        "(placeholder diagram — skeleton pass, to be refined)",
        fill=(120, 120, 120),
        font=small_font,
    )

    # Core loop boxes
    user_box = (40, 120, 260, 180)
    reasoner_box = (460, 120, 700, 180)
    stop_box = (460, 240, 700, 300)
    synth_box = (460, 360, 700, 420)
    answer_box = (900, 360, 1140, 420)

    _box(draw, user_box, "User Request", box_font)
    _box(draw, reasoner_box, "Reasoner\n(plan next action)", box_font)
    _box(draw, stop_box, "Stop?", box_font)
    _box(draw, synth_box, "Synthesizer", box_font)
    _box(draw, answer_box, "Final Answer", box_font)

    _arrow(draw, (260, 150), (460, 150))
    _arrow(draw, (580, 180), (580, 240))
    _arrow(draw, (580, 300), (580, 360))
    _arrow(draw, (700, 390), (900, 390))

    # Tools row
    tools = ["CatalogFilter", "PlotSearch", "SceneSearch", "ExternalContext"]
    tool_y0, tool_y1 = 500, 560
    tool_w = 240
    gap = 40
    start_x = (WIDTH - (tool_w * len(tools) + gap * (len(tools) - 1))) / 2
    tool_boxes = []
    for i, name in enumerate(tools):
        x0 = start_x + i * (tool_w + gap)
        box = (x0, tool_y0, x0 + tool_w, tool_y1)
        tool_boxes.append(box)
        _box(draw, box, name, box_font)

    # Reasoner <-> tools ("Act"/"Observe")
    for box in tool_boxes:
        cx = (box[0] + box[2]) / 2
        _arrow(draw, (580, 180 + 20), (cx, tool_y0 - 40))
    draw.text(
        (start_x, tool_y0 - 40),
        "Act",
        fill=ARROW_COLOR,
        font=small_font,
    )
    for box in tool_boxes:
        cx = (box[0] + box[2]) / 2
        _arrow(draw, (cx, tool_y0), (580, 300 - 10))
    draw.text(
        (start_x + tool_w * 2, tool_y0 - 40),
        "Observe",
        fill=ARROW_COLOR,
        font=small_font,
    )

    draw.text(
        (40, 620),
        "Reasoner loops through Act/Observe against the four tools until Stop?\n"
        "resolves, then Synthesizer composes the Final Answer. Every LLM call\n"
        "along this path is logged in the /api/execute steps trace under the\n"
        "same module name shown here.",
        fill=(90, 90, 90),
        font=small_font,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
