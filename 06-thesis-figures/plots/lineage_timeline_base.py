#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/lineage_timeline_base.py
# Thesis output : base layout and constants of Figure 2 (an earlier draft of the figure is written to staged/)
# Inputs        : none (the timeline content is written in the script)
# Outputs       : staged/fig_1_X_lineage_timeline.{png,pdf}
# Run           : python -P 06-thesis-figures/plots/lineage_timeline_base.py
# Provenance    : adapted from the thesis figure scripts; the drawing is unchanged. Lines marked "# repo path"
#                 read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Base layout of thesis Figure 2 (fig:lineage_timeline, Chapter 1), the encoder and aggregator lineage timeline.

The timeline is drawn as a vector figure on a 2752 x 1536 px canvas (200 dpi -> 13.76 x 7.68 in). The box, arrow,
tick and label positions, the font sizes and the colours follow an earlier raster (PNG) version of the figure:
fonts are Times New Roman (names 16.0 pt, descriptors 14.2 pt, panel titles 16.6 pt, years 16.4 pt), encoder fill
#C4D9EB, aggregator fill #D5D5D5, black outlines, 24 px corner radius.

Changes relative to that earlier version (everything else is reproduced as drawn):
  * the encoder role labels of the earlier version are replaced by descriptors that follow the cited encoder papers:
      UNI      "PathEncoder"  -> "1,024-d features"  (UNI outputs 1024-d feature vectors)
      UNI2-h   "LargeEncoder" -> "1,536-d features"  (UNI2-h: 1536-d, newer HuggingFace variant)
      Virchow  "PathEncoder"  -> "ViT-H/14"          (Virchow: a 632M-parameter ViT-H/14 tile encoder)
      GigaPath "ScaleEncoder" -> "ViT-g/14"          (GigaPath: a ViT-g/14 WSI encoder)
      CONCH    "TextEncoder"  -> "Vision-language"   (CONCH: a vision-language pathology foundation model)
  * UNI2-h moved from 2024 to 2025 (thesis Ch.2: "UNI2-h (Chen et al. 2024, released 2025 update)"). It is
    placed in the 2025 column, bottom-aligned with CONCH, because the UNI -> CLAM-TR dashed arrow crosses the
    upper part of that column; its dashed arrow to CLAM-TR starts from its new right edge. The slot it left in
    the 2024 stack stays empty (the other boxes keep their positions).
  * Spacing: "AttentionMIL" -> "Attention MIL", "NystromTransformer" -> "Nyström Transformer" (spelling of
    thesis Chapter 1), "Double-TierDistillation" -> "Double-Tier Distillation",
    "HierarchicalRetention" -> "Hierarchical Retention". As in the earlier version, a line too wide for its box is
    set in a smaller size (auto-fit with 6 px side padding).
Outputs (subfolder staged/ next to this script): fig_1_X_lineage_timeline.png (2752 x 1536) and
fig_1_X_lineage_timeline.pdf (vector). The two other Figure 2 scripts import this file for its constants and read
the PDF as the reference for their word checks.
Usage: python -P lineage_timeline_base.py
"""
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "staged")
os.makedirs(OUT, exist_ok=True)

W, H, DPI = 2752, 1536, 200
PT = 72.0 / DPI                       # points per pixel
FONT = "Times New Roman"
SIZE_NAME, SIZE_DESC, SIZE_TITLE, SIZE_YEAR = 16.0, 14.2, 16.6, 16.4
ENC_FILL, AGG_FILL = "#C4D9EB", "#D5D5D5"
BORDER_PX, BORDER_THICK_PX, RADIUS_PX = 4.5, 9.0, 24.0
PAD_PX = 6.0
PITCH_PX = 41.0                       # baseline-to-baseline distance inside the boxes
CAP = 0.662                           # Times New Roman cap height (em)

YEAR_X = {2018: 328, 2019: 603, 2020: 878, 2021: 1153, 2022: 1427, 2023: 1702, 2024: 1977, 2025: 2251, 2026: 2527}

# (key, outer x0, x1, y0, y1, fill, lines, bold name, thick border); outer = incl. border, original px
# UNI2-h: original size (262 x 148 px); centred 10 px right of the 2025 tick, the same offset the original
# 2024 stack has from its tick (box centre 1987 vs tick 1977), which leaves a 12 px gap to CONCH.
UNI2H_W, UNI2H_H = 262, 148
UNI2H_XC = YEAR_X[2025] + 10
BOXES = [
    ("UNI", 1856, 2118, 54, 199, ENC_FILL, ["UNI", "ViT-L/16", "1,024-d features"], False, False),
    ("UNI2-h", UNI2H_XC - UNI2H_W / 2, UNI2H_XC + UNI2H_W / 2, 747 - UNI2H_H, 747, ENC_FILL,
     ["UNI2-h", "ViT-H/14", "1,536-d features"], False, False),
    ("Virchow", 1855, 2118, 368, 487, ENC_FILL, ["Virchow", "ViT-H/14"], False, False),
    ("GigaPath", 1856, 2118, 498, 617, ENC_FILL, ["GigaPath", "ViT-g/14"], False, False),
    ("CONCH", 1856, 2118, 628, 747, ENC_FILL, ["CONCH", "Vision-language"], False, False),
    # method descriptor follows the thesis title ("Attention-Enhanced")
    ("CLAM-TR", 2342, 2699, 793, 950, ENC_FILL, ["CLAM-TR", "Attention-Enhanced", "Dual-Branch"], True, True),
    ("ABMIL", 187, 479, 957, 1104, AGG_FILL, ["ABMIL", "Attention MIL", "Aggregator"], False, False),
    ("CLAM", 986, 1306, 786, 939, AGG_FILL, ["CLAM", "Encoder-Agnostic", "MIL"], False, False),
    ("DSMIL", 986, 1306, 957, 1074, AGG_FILL, ["DSMIL", "Dual-Stream MIL"], False, False),
    ("TransMIL", 980, 1306, 1098, 1262, AGG_FILL, ["TransMIL", "Nyström Transformer", "MIL"], False, False),
    ("DTFD-MIL", 1317, 1658, 1098, 1262, AGG_FILL, ["DTFD-MIL", "Double-Tier Distillation", "MIL"], False, False),
    ("RetMIL", 1814, 2159, 957, 1104, AGG_FILL, ["RetMIL", "Hierarchical Retention", "MIL"], False, False),
    ("MamMIL", 1835, 2131, 1114, 1262, AGG_FILL, ["MamMIL", "State-Space MIL"], False, False),
]
BOX = {b[0]: b for b in BOXES}


def right_mid(key):
    b = BOX[key]
    return (b[2], (b[3] + b[4]) / 2.0)


# CLAM-TR top border. The earlier tip was at x ~2515 (box centre 2520); 2532 keeps the UNI line >= 20 px
# clear of the moved UNI2-h box.
DASH_TIP = (2532.0, 791.0)


def head(ax, tip, direction, length, width, notch=0.28, z=5):
    """Filled notched ('stealth') arrow head, tip at `tip`, pointing along `direction`."""
    u = np.asarray(direction, float)
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    t = np.asarray(tip, float)
    base = t - length * u
    pts = [t, base + v * width / 2, t - (1 - notch) * length * u, base - v * width / 2]
    ax.add_patch(Polygon(pts, closed=True, facecolor="black", edgecolor="black", linewidth=0.3, zorder=z))


def line(ax, p0, p1, lw_px, dashed=False, z=1):
    kw = dict(color="black", linewidth=lw_px * PT, zorder=z, solid_capstyle="butt", dash_capstyle="butt")
    if dashed:
        kw["linestyle"] = (0, (21 * PT, 15 * PT))
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **kw)


def fit_text(ax, renderer, x, y, s, size, weight, max_w):
    t = ax.text(x, y, s, ha="center", va="baseline", fontsize=size, family=FONT, weight=weight, zorder=4)
    w = t.get_window_extent(renderer=renderer).width
    if w > max_w:
        t.set_fontsize(size * max_w / w)
    return t


def main():
    mpl.rcdefaults()
    mpl.rcParams["lines.scale_dashes"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    # frame and panel divider
    ax.add_patch(plt.Rectangle((168, 39), 2709 - 168, 1279 - 39, fill=False, edgecolor="black", linewidth=3 * PT, zorder=1))
    ax.plot([168, 2709], [767.5, 767.5], color="black", linewidth=3.5 * PT, zorder=1)

    # panel titles (rotated, reading bottom-to-top)
    for text, xb, yc in (("Foundation-Model", 92, 411.5), ("Encoders (pathology)", 139, 406.5),
                         ("WSI Multiple Instance", 92, 1025.5), ("Learning Aggregators", 139, 1029.5)):
        ax.text(xb, yc, text, rotation=90, rotation_mode="anchor", ha="center", va="baseline",
                fontsize=SIZE_TITLE, family=FONT, zorder=4)

    # time axis, ticks, year labels
    line(ax, (43, 1317), (2680, 1317), 7)
    head(ax, (2708, 1317), (1, 0), 38, 38)
    for yr, x in YEAR_X.items():
        ax.plot([x, x], [1293, 1340], color="black", linewidth=3.5 * PT, zorder=2, solid_capstyle="butt")
        ax.text(x, 1399, str(yr), ha="center", va="baseline", fontsize=SIZE_YEAR, family=FONT, zorder=4)

    # connectors (drawn under the boxes)
    a0, a1 = (479, 1029.6), (986, 861.0)                                   # ABMIL -> CLAM
    line(ax, a0, a1, 3.5)
    head(ax, a1, np.subtract(a1, a0), 30, 30)
    c0, c1 = (1306, 860.4), (2342, 857.0)                                  # CLAM -> CLAM-TR (double head)
    line(ax, c0, (2300, 857.1), 7)
    head(ax, c1, np.subtract(c1, c0), 30, 32)
    head(ax, (2316, 857.1), np.subtract(c1, c0), 30, 32)
    line(ax, (1306, 1011.7), (2296, 857.6), 1.8)                           # DSMIL -> CLAM-TR (thin)
    for key in ("UNI", "UNI2-h"):                                          # encoders -> CLAM-TR (dashed)
        line(ax, right_mid(key), DASH_TIP, 3, dashed=True)
    d_uni = np.subtract(DASH_TIP, right_mid("UNI"))
    d_u2 = np.subtract(DASH_TIP, right_mid("UNI2-h"))
    bis = d_uni / np.linalg.norm(d_uni) + d_u2 / np.linalg.norm(d_u2)
    head(ax, DASH_TIP, bis, 34, 34)

    # boxes and their text
    fitted = []
    for key, x0, x1, y0, y1, fill, lines, bold, thick in BOXES:
        lw = BORDER_THICK_PX if thick else BORDER_PX
        ax.add_patch(FancyBboxPatch((x0 + lw / 2, y0 + lw / 2), (x1 - x0) - lw, (y1 - y0) - lw,
                                    boxstyle=f"round,pad=0,rounding_size={RADIUS_PX - lw / 2}",
                                    facecolor=fill, edgecolor="black", linewidth=lw * PT, zorder=3))
        n = len(lines)
        em1 = SIZE_NAME / PT
        block = CAP * em1 + PITCH_PX * (n - 1)
        b1 = (y0 + y1) / 2.0 - block / 2.0 + CAP * em1
        xc = (x0 + x1) / 2.0
        max_w = (x1 - x0) - 2 * lw - 2 * PAD_PX
        for i, s in enumerate(lines):
            size = SIZE_NAME if i == 0 else SIZE_DESC
            t = fit_text(ax, renderer, xc, b1 + PITCH_PX * i, s, size, "bold" if (bold and i == 0) else "normal", max_w)
            fitted.append((key, s, round(t.get_fontsize(), 2)))

    png = os.path.join(OUT, "fig_1_X_lineage_timeline.png")
    pdf = os.path.join(OUT, "fig_1_X_lineage_timeline.pdf")
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    for key, s, size in fitted:
        if size not in (SIZE_NAME, SIZE_DESC):
            print(f"  auto-fit: {key:9s} '{s}' -> {size} pt")
    print("Wrote", os.path.relpath(png, HERE), "and", os.path.relpath(pdf, HERE))


if __name__ == "__main__":
    main()
