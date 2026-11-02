#!/usr/bin/env python3
"""Create an equivalent electrical circuit (EEC) diagram as SVG and PNG.

The drawing mimics the PowerPoint-drawn EEC.svg in this folder: 1280x720
white canvas, black 3 px strokes, IEC rectangle resistors, three-semicircle
inductor, chevron-pair CPEs, two-plate capacitors, parallel (R,CPE) groups
as rectangular loops, Cambria-Math-style labels.

Layout rules:
  * every series element is connected on one shared horizontal trunk line;
  * a parallel group's vertical connectors are centered on the trunk, so
    the wires enter/leave the group at the midpoint between its branches;
  * a CPE is wired to the apex (middle vertex) of its left chevron on the
    incoming side and to the apex of the right chevron on the outgoing side.

Usage:
    python create_eec_diagram.py "R0-L-(R1,CPE1)" [-o OUTPUT_PREFIX]

Model syntax:
    series elements separated by "-", parallel groups in parentheses with
    branches separated by ",":
        R0-L-(R1,Q1)-(R2,Q2)
    elements: R<n>, L<n>, C<n>, Q<n> for CPE (CPE<n> also accepted; the
    figure label is always "CPE")

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
MARGIN = 150.0
GAP = 70.0
INTRAGAP = 40.0
GROUP_PAD = 30.0
BRANCH_DY = 100.0
Y0 = H / 2
STROKE = 3.0
FONT = 32.0
FONT_SUB = 23.36
LEAF_W = {"R": 100.0, "L": 99.0, "C": 28.0, "CPE": 70.0}
LEFT_PAD = {"R": 50.0, "L": 49.5, "C": 14.0, "CPE": 26.0}
MATH_LETTER = {"R": "\U0001d445", "L": "\U0001d43f", "C": "\U0001d436", "CPE": "CPE"}


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
        elif t == ",":
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
        else:
            m = re.fullmatch(r"([A-Za-z]+)(\d*)", t)
            name = m.group(1).upper()
            if name == "Q":
                name = "CPE"  # Q is the CPE element; the figure label stays "CPE"
            if name not in MATH_LETTER:
                raise ValueError(f"Unknown element {t!r} (use R, L, C, Q/CPE)")
            series.append(("leaf", name, m.group(2)))
            pos += 1
    return series, pos


def group_columns(branches, s: float) -> list[float]:
    """Column layout shared by all branches: element i of every branch gets the
    same width (the max over branches), so elements align vertically."""
    ncols = max(len(b) for b in branches)
    cols = []
    for c in range(ncols):
        w = max((LEAF_W[b[c][1]] for b in branches if c < len(b)), default=0.0)
        cols.append(w * s)
    return cols


def item_width(item, s: float) -> float:
    if item[0] == "leaf":
        return LEAF_W[item[1]] * s
    cols = group_columns(item[1], s)
    return sum(cols) + INTRAGAP * s * max(0, len(cols) - 1) + 2 * GROUP_PAD * s


def layout(series) -> list[tuple]:
    """Build a primitive list with absolute coordinates. Primitives:
    ('line', x1, y1, x2, y2), ('poly', [(x, y), ...]), ('rect', cx, cy, w, h),
    ('arc', cx, cy, r) upper semicircle, ('cap', cx, cy, h, gap) two vertical
    plates, ('text', x, y, main, sub, fs), ('stroke', width)
    """
    needed = sum(item_width(it, 1.0) for it in series) + GAP * max(0, len(series) - 1)
    s = min(1.0, (W - 2 * MARGIN) / needed) if needed else 1.0

    prims: list[tuple] = []

    def wire(x1, x2, y):
        if x2 > x1:
            prims.append(("line", x1, y, x2, y))

    def label(x, y, name, num, below=False):
        ty = y + 68 * s if below else y - 72 * s
        prims.append(("text", x, ty, name, num, FONT * s))

    def draw_leaf(kind, left, cy):
        """Draw a component whose symbol starts at x=left on wire level cy.
        Return (left_conn, right_conn): x positions where wires attach."""
        cx = left + LEFT_PAD[kind] * s
        if kind == "R":
            prims.append(("rect", cx, cy, 100 * s, 52 * s))
            return cx - 50 * s, cx + 50 * s
        if kind == "L":
            r = 16.5 * s
            for i in range(3):
                prims.append(("arc", cx + (i - 1) * 2 * r, cy, r))
            return cx - 3 * r, cx + 3 * r
        if kind == "C":
            prims.append(("cap", cx, cy, 80 * s, 28 * s))
            return cx - 14 * s, cx + 14 * s
        for i in range(2):
            x0 = cx - 26 * s + 18 * s * i
            prims.append(("poly", [(x0, cy - 40 * s), (x0 + 26 * s, cy), (x0, cy + 40 * s)]))
        return cx, cx + 18 * s

    def draw_branch(branch, x0, x1, y, cols, label_below):
        cursor = x0
        prev_conn = None
        for c, (_, kind, num) in enumerate(branch):
            w = cols[c]
            left = cursor + (w - LEAF_W[kind] * s) / 2
            l, r = draw_leaf(kind, left, y)
            wire(x0 if prev_conn is None else prev_conn, l, y)
            label(cursor + w / 2, y, kind, num, below=label_below)
            prev_conn = r
            cursor += w + INTRAGAP * s
        wire(prev_conn, x1, y)

    x = MARGIN
    prev_end = MARGIN - 70
    for item in series:
        w = item_width(item, s)
        wire(prev_end, x, Y0)
        if item[0] == "leaf":
            kind, num = item[1], item[2]
            l, r = draw_leaf(kind, x, Y0)
            wire(x, l, Y0)
            wire(r, x + w, Y0)
            label(x + LEAF_W[kind] * s / 2, Y0, kind, num)
        else:
            branches = item[1]
            k = len(branches)
            cols = group_columns(branches, s)
            ys = [Y0 + (i - (k - 1) / 2) * BRANCH_DY * s for i in range(k)]
            for i, branch in enumerate(branches):
                draw_branch(branch, x, x + w, ys[i], cols, label_below=(k > 1 and i == k - 1))
            if k > 1:
                ymid = Y0
                prims.append(("line", x, ys[0], x, ymid))
                prims.append(("line", x, ymid, x, ys[-1]))
                prims.append(("line", x + w, ys[0], x + w, ymid))
                prims.append(("line", x + w, ymid, x + w, ys[-1]))
        prev_end = x + w
        x += w + GAP * s
    wire(prev_end, prev_end + 70, Y0)
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
        elif p[0] == "poly":
            d = f"M{p[1][0][0]:.3f} {p[1][0][1]:.3f} " + " ".join(
                f"L{x:.3f} {y:.3f}" for x, y in p[1][1:]
            )
            parts.append(f'<path d="{d}" {style}/>')
        elif p[0] == "rect":
            _, cx, cy, w, h = p
            parts.append(
                f'<rect x="{cx - w / 2:.3f}" y="{cy - h / 2:.3f}" width="{w:.3f}" height="{h:.3f}" '
                f'stroke="#000000" stroke-width="{stroke}" stroke-miterlimit="8" fill="none"/>'
            )
        elif p[0] == "arc":
            _, cx, cy, r = p
            parts.append(f'<path d="M{cx - r:.3f} {cy:.3f} A{r:.3f} {r:.3f} 0 0 1 {cx + r:.3f} {cy:.3f}" {style}/>')
        elif p[0] == "cap":
            _, cx, cy, h, gap = p
            for dx in (-gap / 2, gap / 2):
                parts.append(f'<path d="M{cx + dx:.3f} {cy - h / 2:.3f} {cx + dx:.3f} {cy + h / 2:.3f}" {style}/>')
        elif p[0] == "text":
            _, tx, ty, main, sub, fs = p
            glyph = MATH_LETTER.get(main, main)
            sub_html = (
                f'<tspan font-size="{FONT_SUB * fs / FONT:.2f}" dy="{7 * fs / FONT:.2f}">{sub}</tspan>'
                if sub
                else ""
            )
            parts.append(
                f'<text font-family="Cambria Math,Cambria Math_MSFontService,sans-serif" font-weight="400" '
                f'font-size="{fs:.2f}" text-anchor="middle" transform="translate({tx:.3f} {ty:.3f})">'
                f"{glyph}{sub_html}</text>"
            )
    parts.append("</g></svg>")
    return "".join(parts)


def render_png(prims, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["mathtext.fontset"] = "cm"
    stroke = next((p[1] for p in prims if p[0] == "stroke"), STROKE)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def plot(xs, ys):
        ax.plot(xs, ys, color="black", lw=stroke, solid_joinstyle="miter", solid_capstyle="butt")

    for p in prims:
        if p[0] == "line":
            _, x1, y1, x2, y2 = p
            plot([x1, x2], [y1, y2])
        elif p[0] == "poly":
            for a, b in zip(p[1], p[1][1:]):
                plot([a[0], b[0]], [a[1], b[1]])
        elif p[0] == "rect":
            _, cx, cy, w, h = p
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="black", lw=stroke))
        elif p[0] == "arc":
            _, cx, cy, r = p
            t = [i * math.pi / 60 for i in range(61)]
            plot([cx + r * math.cos(v) for v in t], [cy - r * math.sin(v) for v in t])
        elif p[0] == "cap":
            _, cx, cy, h, gap = p
            for dx in (-gap / 2, gap / 2):
                plot([cx + dx, cx + dx], [cy - h / 2, cy + h / 2])
        elif p[0] == "text":
            _, tx, ty, main, sub, fs = p
            if main == "CPE":
                txt = rf"$\mathrm{{CPE}}_{{{sub}}}$" if sub else r"$\mathrm{CPE}$"
            else:
                txt = rf"${main}_{{{sub}}}$" if sub else rf"${main}$"
            ax.text(tx, ty, txt, fontsize=fs * 72 / 100, ha="center", va="baseline", color="black")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create an EEC diagram as SVG and PNG.")
    ap.add_argument("model", help='EEC model, e.g. "R0-L-(R1,Q1)"')
    ap.add_argument("-o", "--out", default=None, help="output path prefix (default: EEC_<model> next to this script)")
    args = ap.parse_args()

    toks = tokenize(args.model)
    series, pos = parse(toks)
    if pos != len(toks) or not series:
        raise ValueError(f"Cannot parse model string: {args.model!r}")
    prims = layout(series)

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
