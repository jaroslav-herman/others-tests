#!/usr/bin/env python3
"""Create an equivalent electrical circuit (EEC) diagram as SVG and PNG.

The drawing mimics the PowerPoint-drawn EEC.svg in this folder: 1280x720
white canvas, black 3 px strokes, IEC rectangle resistors, semicircle-arc
inductor, chevron-pair CPEs, parallel (R,CPE) groups as rectangular loops
around the main wire, Cambria-Math-style labels.

Usage:
    python create_eec_diagram.py "R0-L-(R1,CPE1)" [-o OUTPUT_PREFIX]

Model syntax:
    series elements separated by "-", parallel groups in parentheses with
    branches separated by ",":
        R0-L-(R1,CPE1)-(R2,CPE2)
    elements: R<n>, L<n>, C<n>, CPE<n> (Q<n> accepted as CPE alias)

Outputs OUTPUT_PREFIX.svg and OUTPUT_PREFIX.png (default prefix:
EEC_<sanitized model> next to this script). PNG export needs matplotlib.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

W, H = 1280, 720
MARGIN = 150
GAP = 70
LEAF_W = 170
GROUP_W = 240
BRANCH_DY = 100
STROKE = 3.0
FONT = 32.0
FONT_SUB = 23.36
MATH_LETTER = {"R": "\U0001d445", "L": "\U0001d43f", "C": "\U0001d436", "CPE": "CPE", "Q": "CPE"}


def tokenize(model: str) -> list[str]:
    toks = re.findall(r"[A-Za-z]+\d*|[(),\-]", model)
    if not toks or "".join(toks) != re.sub(r"\s+", "", model):
        raise ValueError(f"Cannot parse model string: {model!r}")
    return toks


def parse(toks: list[str], pos: int = 0):
    """Parse into a series list; each item is ('leaf', name, num) or ('group', [branch, ...])."""
    series = []
    while pos < len(toks):
        t = toks[pos]
        if t == "-":
            pos += 1
        elif t == ")":
            break
        elif t == "(":
            pos += 1
            branches = []
            while True:
                branch, pos = parse(toks, pos)
                branches.append(branch)
                if pos < len(toks) and toks[pos] == ",":
                    pos += 1
                elif pos < len(toks) and toks[pos] == ")":
                    pos += 1
                    break
                else:
                    raise ValueError(f"Expected ',' or ')' at token {pos}")
            series.append(("group", branches))
        elif t == ",":
            break
        else:
            m = re.fullmatch(r"([A-Za-z]+)(\d*)", t)
            name = m.group(1).upper()
            if name == "Q":
                name = "CPE"
            if name not in MATH_LETTER:
                raise ValueError(f"Unknown element {t!r} (use R, L, C, CPE)")
            series.append(("leaf", name, m.group(2)))
            pos += 1
    return series, pos


def item_width(item) -> float:
    if item[0] == "leaf":
        return LEAF_W
    return max(GROUP_W, max(len(b) for b in item[1]) * LEAF_W)


def layout(series) -> list[tuple]:
    """Build a primitive list. Primitives:
    ('line', x1, y1, x2, y2), ('rect', cx, cy, w, h), ('arc', cx, cy, r),
    ('chev', cx, cy, w, h), ('cap', cx, cy, h, gap), ('text', x, y, main, sub, fs)
    """
    kmax = max([1] + [len(it[1]) for it in series if it[0] == "group"])
    needed = sum(item_width(it) for it in series) + GAP * max(0, len(series) - 1)
    s = min(1.0, (W - 2 * MARGIN) / needed) if needed else 1.0
    y0 = 370 - (kmax - 1) * 50

    prims: list[tuple] = []

    def wire(x1, y1, x2, y2):
        if x2 > x1:
            prims.append(("line", x1, y1, x2, y2))

    def label(cx, cy, name, num, above=True):
        main = MATH_LETTER[name]
        prims.append(("text", cx, cy - 72 * s if above else cy + 68 * s, main, num, FONT * s))

    def draw_leaf(kind, cx, cy):
        """Draw component centered on the wire; return (left edge, right edge) of the break."""
        if kind == "R":
            w, h = 100 * s, 52 * s
            prims.append(("rect", cx, cy, w, h))
            return cx - w / 2, cx + w / 2
        if kind == "L":
            r = 16.5 * s
            for i in range(3):
                prims.append(("arc", cx + (i - 1) * 2 * r, cy, r))
            return cx - 3 * r, cx + 3 * r
        if kind == "C":
            prims.append(("cap", cx, cy, 80 * s, 28 * s))
            return cx - 14 * s, cx + 14 * s
        # CPE
        prims.append(("chev", cx, cy, 26 * s, 80 * s))
        return cx - 30 * s, cx + 14 * s

    def draw_branch(leaves, x0, x1, y, label_above):
        n = len(leaves)
        step = (x1 - x0) / n
        prev_edge = x0
        for i, (_, name, num) in enumerate(leaves):
            cx = x0 + step * (i + 0.5)
            le, ri = draw_leaf(name, cx, y)
            wire(prev_edge, y, le, y)
            wire(ri, y, x0 + step * (i + 1), y)
            label(cx, y, name, num, above=label_above)
            prev_edge = ri
        wire(prev_edge, y, x1, y)

    x = float(MARGIN)
    for item in series:
        w = item_width(item) * s
        if item[0] == "leaf":
            cx = x + w / 2
            le, ri = draw_leaf(item[1], cx, y0)
            wire(x, y0, le, y0)
            wire(ri, y0, x + w, y0)
            label(cx, y0, item[1], item[2], above=True)
        else:
            branches = item[1]
            for i, branch in enumerate(branches):
                draw_branch(branch, x, x + w, y0 + i * BRANCH_DY * s, label_above=(i == 0))
            if len(branches) > 1:
                prims.append(("line", x, y0, x, y0 + (len(branches) - 1) * BRANCH_DY * s))
                prims.append(("line", x + w, y0, x + w, y0 + (len(branches) - 1) * BRANCH_DY * s))
            else:
                wire(x, y0, x + w, y0)
        x += w + GAP * s

    wire(MARGIN - 70, y0, MARGIN, y0)
    wire(x - GAP * s, y0, x - GAP * s + 70, y0)
    prims.append(("stroke", STROKE * s))
    return prims


def render_svg(prims) -> str:
    stroke = next((p[1] for p in prims if p[0] == "stroke"), STROKE)
    parts = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" overflow="hidden"><g>'
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>'
    ]
    style = f'stroke="#000000" stroke-width="{stroke}" stroke-miterlimit="8" fill="none" fill-rule="evenodd"'
    for p in prims:
        if p[0] == "line":
            _, x1, y1, x2, y2 = p
            parts.append(f'<path d="M{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}" {style}/>')
        elif p[0] == "rect":
            _, cx, cy, w, h = p
            parts.append(
                f'<rect x="{cx - w / 2:.3f}" y="{cy - h / 2:.3f}" width="{w:.3f}" height="{h:.3f}" '
                f'stroke="#000000" stroke-width="{stroke}" stroke-miterlimit="8" fill="none"/>'
            )
        elif p[0] == "arc":
            _, cx, cy, r = p
            parts.append(f'<path d="M{cx - r:.3f} {cy:.3f} A{r:.3f} {r:.3f} 0 0 1 {cx + r:.3f} {cy:.3f}" {style}/>')
        elif p[0] == "chev":
            _, cx, cy, w, h = p
            for i in range(2):
                dx = 18 * s_global * i
                parts.append(
                    f'<path d="M{cx - 30 * s_global + dx:.3f} {cy - h / 2:.3f} {cx - 4 * s_global + dx:.3f} {cy:.3f} '
                    f'{cx - 30 * s_global + dx:.3f} {cy + h / 2:.3f}" {style}/>'
                )
        elif p[0] == "cap":
            _, cx, cy, h, gap = p
            for dx in (-gap / 2, gap / 2):
                parts.append(f'<path d="M{cx + dx:.3f} {cy - h / 2:.3f} {cx + dx:.3f} {cy + h / 2:.3f}" {style}/>')
        elif p[0] == "text":
            _, tx, ty, main, sub, fs = p
            sub_html = f'<tspan font-size="{FONT_SUB * fs / FONT:.2f}" dy="{7 * fs / FONT:.2f}">{sub}</tspan>' if sub else ""
            parts.append(
                f'<text font-family="Cambria Math,Cambria Math_MSFontService,sans-serif" font-weight="400" '
                f'font-size="{fs:.2f}" text-anchor="middle" transform="translate({tx:.3f} {ty:.3f})">{main}{sub_html}</text>'
            )
    parts.append("</g></svg>")
    return "".join(parts)


s_global = 1.0


def render_png(prims, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["mathtext.fontset"] = "cm"
    stroke = next((p[1] for p in prims if p[0] == "stroke"), STROKE)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for p in prims:
        if p[0] == "line":
            _, x1, y1, x2, y2 = p
            ax.plot([x1, x2], [y1, y2], color="black", lw=stroke, solid_joinstyle="miter", solid_capstyle="butt")
        elif p[0] == "rect":
            from matplotlib.patches import Rectangle

            _, cx, cy, w, h = p
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="black", lw=stroke))
        elif p[0] == "arc":
            _, cx, cy, r = p
            t = [i * math.pi / 60 for i in range(61)]
            ax.plot([cx + r * math.cos(v) for v in t], [cy - r * math.sin(v) for v in t],
                    color="black", lw=stroke, solid_joinstyle="miter")
        elif p[0] == "chev":
            _, cx, cy, w, h = p
            for i in range(2):
                dx = w * 18 / 26 * i
                ax.plot([cx - w * 30 / 26 + dx, cx - w * 4 / 26 + dx, cx - w * 30 / 26 + dx],
                        [cy - h / 2, cy, cy + h / 2], color="black", lw=stroke, solid_joinstyle="miter")
        elif p[0] == "cap":
            _, cx, cy, h, gap = p
            for dx in (-gap / 2, gap / 2):
                ax.plot([cx + dx, cx + dx], [cy - h / 2, cy + h / 2], color="black", lw=stroke)
        elif p[0] == "text":
            _, tx, ty, main, sub, fs = p
            pt = fs * 72 / 100
            if main == "CPE":
                txt = rf"$\mathrm{{CPE}}_{{{sub}}}$" if sub else r"$\mathrm{CPE}$"
            else:
                letter = {"\U0001d445": "R", "\U0001d43f": "L", "\U0001d436": "C"}[main]
                txt = rf"${letter}_{{{sub}}}$" if sub else rf"${letter}$"
            ax.text(tx, ty, txt, fontsize=pt, ha="center", va="center", color="black")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create an EEC diagram as SVG and PNG.")
    ap.add_argument("model", help='EEC model, e.g. "R0-L-(R1,CPE1)"')
    ap.add_argument("-o", "--out", default=None, help="output path prefix (default: EEC_<model> next to this script)")
    args = ap.parse_args()

    global s_global
    series, pos = parse(tokenize(args.model))
    if pos != len(tokenize(args.model)):
        raise ValueError("Trailing tokens in model string")
    prims = layout(series)
    s_global = next((p[1] for p in prims if p[0] == "stroke"), STROKE) / STROKE

    prefix = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "EEC_" + re.sub(r"[^A-Za-z0-9]+", "_", args.model).strip("_")
    )
    svg_path = prefix + ".svg"
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(render_svg(prims))
    print(f"Wrote {svg_path}")
    png_path = prefix + ".png"
    try:
        render_png(prims, png_path)
        print(f"Wrote {png_path}")
    except ImportError:
        print("matplotlib not installed; skipped PNG export (pip install matplotlib)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
