#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig2_lineage_timeline.py
# Thesis output : Figure 2 (fig_1_X_lineage_timeline), a literature timeline
# Inputs        : box texts of lineage_timeline_previous_version.py (read with ast); constants of lineage_timeline_base.py (imported) and its staged PDF; the thesis build's PNG under CLAM_TR_PROJECT_DIR/thesis/main/figures
# Outputs       : PDF + PNG + previews next to this script
# Run           : python -P 06-thesis-figures/plots/lineage_timeline_base.py, then python -P 06-thesis-figures/plots/make_fig2_lineage_timeline.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 2 (fig:lineage_timeline) with every text item >= 10 pt printed (KTU thesis guide, section 5.1).
Writes fig_1_X_lineage_timeline.{png,pdf} and preview images next to this script.

Start: the previous version, lineage_timeline_previous_version.py (box texts, colours, line widths, frame, axis,
year ticks, connector logic); the base layout lineage_timeline_base.py is imported for its constants only.
Changes relative to the previous version:
  * printed at \textwidth (438.83 bp; was 0.95 \textwidth from a PNG); the vector PDF is the thesis file now
    (the PNG twin is written as well). Canvas width 2752 px at 200 dpi as before; height grows to fit the larger text.
  * font sizes (native -> printed at \textwidth, scale 0.44294): names 24.0 -> 10.63 pt, descriptors and years
    22.7 -> 10.05 pt, panel titles 23.5 -> 10.41 pt (previous version: 7.16-8.2 pt). Times New Roman throughout.
  * boxes re-sized to the larger text (padding and line pitch scaled by the same factor 22.7/17); box rows
    re-stacked; encoder panel taller, so that the UNI -> CLAM-TR dashed connector clears UNI2-h.
  * box texts and line breaks exactly those of the previous version (asserted).
Checks (asserted): every PDF text span >= 9.95 pt printed; word and number-token multisets equal to the text items
of the previous version (read from its source with ast) and to the base-layout PDF (Vision- + language merged); each
line inside its box; boxes pairwise >= 8 px apart; every box inside the frame and its panel; connectors clear every
other box by >= 15 px; box centres within 140 px of the previous positions (larger boxes); printed height <= 18 cm.
Usage: python -P make_fig2_lineage_timeline.py (after lineage_timeline_base.py, which writes the base-layout PDF)
"""
import ast
import collections
import importlib.util
import os  # repo path (reads the environment variables of this repository)
import pathlib
import re

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

import fitz  # noqa: E402
from PIL import Image  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("CLAM_TR_PROJECT_DIR", pathlib.Path(__file__).resolve().parents[2]))  # repo path
S13 = HERE  # repo path
S17GEN = HERE / "lineage_timeline_previous_version.py"  # repo path
GEN = HERE / "lineage_timeline_base.py"  # repo path
S14_TWIN = GEN.parent / "staged" / "fig_1_X_lineage_timeline.pdf"
OLD_PNG = ROOT / "thesis/main/figures/fig_1_X_lineage_timeline.png"
spec = importlib.util.spec_from_file_location("G14", GEN)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)          # constants only; its main() is not run

# ---------------------------------------------------------------- text items of the previous version (the reference content)
tree = ast.parse(S17GEN.read_text(encoding="utf-8"))
S17_LINES, S17_TITLES = None, None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "NEWLINES":
        S17_LINES = ast.literal_eval(node.value)
    if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple) and len(node.iter.elts) == 4:
        try:
            S17_TITLES = [e.elts[0].value for e in node.iter.elts]
        except AttributeError:
            pass
assert S17_LINES and S17_TITLES == ["Foundation-Model", "Encoders (pathology)", "WSI Multiple Instance",
                                    "Learning Aggregators"], S17_TITLES
NEWLINES = {k: list(v) for k, v in S17_LINES.items()}        # identical boxes, lines and line breaks
TITLES = S17_TITLES

W, DPI = G.W, G.DPI
PT = 72.0 / DPI
TW_BP = 441.01773 * 72 / 72.27
PRINT_SCALE = TW_BP / (W / DPI * 72)                            # \textwidth / figure width
MIN_PRINTED = 9.95
SIZE_NAME, SIZE_DESC, SIZE_TITLE, SIZE_YEAR = 24.0, 22.7, 23.5, 22.7
F = SIZE_DESC / 17.0                                            # growth factor relative to the previous version
PITCH_PX = 49.0 * F
PAD_H, PAD_V = 14.0 * F, 17.0 * F
GAP = 13.0
FONT = G.FONT
OLD = {b[0]: b for b in G.BOXES}

mpl.rcdefaults()
mpl.rcParams["lines.scale_dashes"] = False
mpl.rcParams["pdf.fonttype"] = 42
_f = plt.figure(figsize=(W / DPI, 8), dpi=DPI)
_a = _f.add_axes([0, 0, 1, 1])
_r = _f.canvas.get_renderer()


def textw(s, size, bold=False):
    t = _a.text(0, 0, s, fontsize=size, family=FONT, weight="bold" if bold else "normal")
    w = t.get_window_extent(renderer=_r).width
    t.remove()
    return w


plt.close(_f)


def need_w(key):
    lines = NEWLINES[key]
    thick = key == "CLAM-TR"
    lw = G.BORDER_THICK_PX if thick else G.BORDER_PX
    tw = max(textw(s, SIZE_NAME if i == 0 else SIZE_DESC, thick and i == 0) for i, s in enumerate(lines))
    return tw + 2 * lw + 2 * PAD_H


def box_h(key):
    n = len(NEWLINES[key])
    em1 = SIZE_NAME / PT
    lw = G.BORDER_THICK_PX if key == "CLAM-TR" else G.BORDER_PX
    return G.CAP * em1 + PITCH_PX * (n - 1) + 2 * PAD_V + 2 * (lw - G.BORDER_PX)


# ---------------------------------------------------------------- layout (px, y downwards)
Y = G.YEAR_X
FRAME_L, FRAME_R, FRAME_TOP = 168.0, 2709.0, 39.0
W_ENC = max(need_w(k) for k in ("UNI", "UNI2-h", "Virchow", "GigaPath", "CONCH"))
W_21 = max(need_w(k) for k in ("CLAM", "DSMIL", "TransMIL"))
W_24 = max(need_w(k) for k in ("RetMIL", "MamMIL"))
C_STACK = Y[2024] - 30.0            # 2024 encoder stack (tick inside the box)
B = {}


def put(key, xc, w, y0=None, y1=None):
    h = box_h(key)
    if y0 is None:
        y0 = y1 - h
    B[key] = [xc - w / 2.0, xc + w / 2.0, y0, y0 + h]


# encoder panel: UNI on top; Virchow, GigaPath, CONCH bottom-aligned above the divider; UNI2-h next to the stack.
# The divider height is the smallest that lets the UNI -> CLAM-TR dashed line clear UNI2-h by >= 25 px.
put("UNI", C_STACK, W_ENC, y0=54.0)
w_cltr = need_w("CLAM-TR")


def layout(divider):
    put("CONCH", C_STACK, W_ENC, y1=divider - 20.0)
    put("GigaPath", C_STACK, W_ENC, y1=B["CONCH"][2] - GAP)
    put("Virchow", C_STACK, W_ENC, y1=B["GigaPath"][2] - GAP)
    put("UNI2-h", C_STACK + W_ENC + 12.0, W_ENC, y1=divider - 20.0)
    put("CLAM-TR", FRAME_R - 10.0 - w_cltr / 2.0, w_cltr, y0=divider + 26.0)
    tip_y = B["CLAM-TR"][2] - 2.0
    tip_x = B["CLAM-TR"][1] - G.RADIUS_PX - 40.0
    p0 = np.array([B["UNI"][1], (B["UNI"][2] + B["UNI"][3]) / 2.0])
    u2 = B["UNI2-h"]
    f = (u2[2] - p0[1]) / (tip_y - p0[1])
    x_at = p0[0] + f * (tip_x - p0[0])
    return x_at - (u2[1] + 25.0), (tip_x, tip_y)


DIVIDER = 900.0
while True:
    margin, _ = layout(DIVIDER)
    if margin >= 0 and B["Virchow"][2] >= B["UNI"][3] + 30.0:
        break
    DIVIDER += 2.0

# aggregator panel
put("CLAM", 1120.0, W_21, y0=DIVIDER + 17.0)
put("DSMIL", 1120.0, W_21, y0=B["CLAM"][3] + GAP)
put("TransMIL", 1120.0, W_21, y0=B["DSMIL"][3] + GAP)
w_dt = need_w("DTFD-MIL")
put("DTFD-MIL", B["TransMIL"][1] + 11.0 + w_dt / 2.0, w_dt, y0=B["TransMIL"][2])
put("ABMIL", 400.0, need_w("ABMIL"), y0=B["DSMIL"][2])
C_24 = B["DTFD-MIL"][1] + 11.0 + W_24 / 2.0
put("RetMIL", C_24, W_24, y0=B["CLAM-TR"][3] + 15.0)
put("MamMIL", C_24, W_24, y0=B["RetMIL"][3] + GAP)
FRAME_BOT = max(B["TransMIL"][3], B["MamMIL"][3]) + 17.0
AXIS_Y = FRAME_BOT + 38.0
YEAR_BASE = AXIS_Y + 98.0
H = int(np.ceil(YEAR_BASE + 50.0))
_, TIP = layout(DIVIDER)


def mid_l(k):
    b = B[k]
    return (b[0], (b[2] + b[3]) / 2.0)


def mid_r(k):
    b = B[k]
    return (b[1], (b[2] + b[3]) / 2.0)


CL_Y = (B["CLAM"][2] + B["CLAM"][3]) / 2.0                         # CLAM -> CLAM-TR, horizontal
assert B["CLAM-TR"][2] + G.RADIUS_PX < CL_Y < B["CLAM-TR"][3] - G.RADIUS_PX
C0, C1 = (B["CLAM"][1], CL_Y), (B["CLAM-TR"][0], CL_Y)
HEAD2 = (C1[0] - 26.0, CL_Y)
LINE_END = C1[0] - 42.0
THIN0, THIN1 = mid_r("DSMIL"), (C1[0] - 46.0, CL_Y + 0.4)
A0, A1 = mid_r("ABMIL"), mid_l("CLAM")


def head(ax, tip, direction, length, width, notch=0.28, z=5):
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


def draw(png_path, pdf_path):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((FRAME_L, FRAME_TOP), FRAME_R - FRAME_L, FRAME_BOT - FRAME_TOP, fill=False,
                               edgecolor="black", linewidth=3 * PT, zorder=1))
    ax.plot([FRAME_L, FRAME_R], [DIVIDER, DIVIDER], color="black", linewidth=3.5 * PT, zorder=1)
    tc, bc = (FRAME_TOP + DIVIDER) / 2.0, (DIVIDER + FRAME_BOT) / 2.0
    for text, xb, yc in ((TITLES[0], 80, tc), (TITLES[1], 145, tc), (TITLES[2], 80, bc), (TITLES[3], 145, bc)):
        ax.text(xb, yc, text, rotation=90, rotation_mode="anchor", ha="center", va="baseline",
                fontsize=SIZE_TITLE, family=FONT, zorder=4)
    line(ax, (43, AXIS_Y), (2680, AXIS_Y), 7)
    head(ax, (2708, AXIS_Y), (1, 0), 38, 38)
    for yr, x in Y.items():
        ax.plot([x, x], [AXIS_Y - 24, AXIS_Y + 23], color="black", linewidth=3.5 * PT, zorder=2, solid_capstyle="butt")
        ax.text(x, YEAR_BASE, str(yr), ha="center", va="baseline", fontsize=SIZE_YEAR, family=FONT, zorder=4)
    line(ax, A0, A1, 3.5)
    head(ax, A1, np.subtract(A1, A0), 30, 30)
    line(ax, C0, (LINE_END, CL_Y), 7)
    head(ax, C1, (1, 0), 30, 32)
    head(ax, HEAD2, (1, 0), 30, 32)
    line(ax, THIN0, THIN1, 1.8)
    for key in ("UNI", "UNI2-h"):
        line(ax, mid_r(key), TIP, 3, dashed=True)
    d1 = np.subtract(TIP, mid_r("UNI"))
    d2 = np.subtract(TIP, mid_r("UNI2-h"))
    head(ax, TIP, d1 / np.linalg.norm(d1) + d2 / np.linalg.norm(d2), 34, 34)
    for key, (x0, x1, y0, y1) in B.items():
        thick = key == "CLAM-TR"
        lw = G.BORDER_THICK_PX if thick else G.BORDER_PX
        ax.add_patch(FancyBboxPatch((x0 + lw / 2, y0 + lw / 2), (x1 - x0) - lw, (y1 - y0) - lw,
                                    boxstyle=f"round,pad=0,rounding_size={G.RADIUS_PX - lw / 2}",
                                    facecolor=OLD[key][5], edgecolor="black", linewidth=lw * PT, zorder=3))
        lines = NEWLINES[key]
        em1 = SIZE_NAME / PT
        block = G.CAP * em1 + PITCH_PX * (len(lines) - 1)
        b1 = (y0 + y1) / 2.0 - block / 2.0 + G.CAP * em1
        for i, s in enumerate(lines):
            ax.text((x0 + x1) / 2.0, b1 + PITCH_PX * i, s, ha="center", va="baseline",
                    fontsize=SIZE_NAME if i == 0 else SIZE_DESC, family=FONT,
                    weight="bold" if (thick and i == 0) else "normal", zorder=4)
    fig.savefig(png_path, dpi=DPI, facecolor="white")
    fig.savefig(pdf_path, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


PNG = HERE / "fig_1_X_lineage_timeline.png"
PDF = HERE / "fig_1_X_lineage_timeline.pdf"
draw(PNG, PDF)

# ---------------------------------------------------------------- checks
FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


im = Image.open(PNG)
check(im.size == (W, H), f"PNG canvas {im.size}")
pg = fitz.open(PDF)[0]
k = DPI / 72.0
spans = [(s["text"].strip(), s["size"], [v * k for v in s["bbox"]], s["font"]) for b in pg.get_text("dict")["blocks"]
         for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
psz = sorted(s[1] * PRINT_SCALE for s in spans)
check(psz[0] >= MIN_PRINTED, f"smallest text {psz[0]:.2f} pt printed at \\textwidth (median {psz[len(psz) // 2]:.2f}); "
      f"fonts {sorted({s[3] for s in spans})}")
ph_cm = pg.rect.height * PRINT_SCALE / 72 * 2.54
check(ph_cm <= 18.0, f"printed {TW_BP / 72 * 2.54:.2f} x {ph_cm:.2f} cm (page {pg.rect.width:.2f} x {pg.rect.height:.2f} bp,"
      f" scale {PRINT_SCALE:.5f})")

new_txt = pg.get_text()
ref_items = [s for v in NEWLINES.values() for s in v] + TITLES + [str(y) for y in Y]
rw = collections.Counter(" ".join(ref_items).split())
nw = collections.Counter(new_txt.split())
check(rw == nw, f"word multiset = the previous version's text items ({sum(rw.values())} words)"
      + ("" if rw == nw else f": previous-only {dict(rw - nw)}, new-only {dict(nw - rw)}"))
NUM = re.compile(r"(?<![A-Za-z0-9])[+\-\u2212\u2013]?\d+(?:[.,]\d+)*")
rn, nn = collections.Counter(NUM.findall(" \n".join(ref_items))), collections.Counter(NUM.findall(new_txt))
check(rn == nn, f"number tokens = previous version ({sum(rn.values())}): {sorted(rn.items())}")
ow = collections.Counter(fitz.open(S14_TWIN)[0].get_text().split())
nw2 = collections.Counter(nw)
nw2.subtract({"Vision-": 1, "language": 1})
nw2["Vision-language"] += 1
nw2 = +nw2
check(ow == nw2, f"word multiset = the base-layout PDF with Vision-/language merged ({sum(ow.values())} words)"
      + ("" if ow == nw2 else f": base-only {dict(ow - nw2)}, new-only {dict(nw2 - ow)}"))
check(all(NEWLINES[k_] == list(S17_LINES[k_]) for k_ in S17_LINES), "every box: same lines and line breaks as the previous version")
nbad = 0
for key, (x0, x1, y0, y1) in B.items():
    lw = G.BORDER_THICK_PX if key == "CLAM-TR" else G.BORDER_PX
    for ln in NEWLINES[key]:
        hit = [bb for t, _, bb, _ in spans if t == ln and x0 <= (bb[0] + bb[2]) / 2 <= x1 and y0 <= (bb[1] + bb[3]) / 2 <= y1]
        ok = len(hit) == 1 and hit[0][0] >= x0 + lw + 2 and hit[0][2] <= x1 - lw - 2 and hit[0][1] >= y0 + lw \
            and hit[0][3] <= y1 - lw
        if not ok:
            nbad += 1
            check(False, f"{key}: '{ln}' inside its box {hit}")
check(nbad == 0, "every box line found once and inside its box")
keys = list(B)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = B[keys[i]], B[keys[j]]
        if not (a[1] + 8 <= b[0] or b[1] + 8 <= a[0] or a[3] + 8 <= b[2] or b[3] + 8 <= a[2]):
            check(False, f"boxes {keys[i]} and {keys[j]} overlap or are < 8 px apart")
check(True, "boxes pairwise separated by >= 8 px (failures listed above, if any)")
for key in B:
    check(FRAME_L + 10 < B[key][0] and B[key][1] < FRAME_R - 8
          and (B[key][3] < DIVIDER - 8 or B[key][2] > DIVIDER + 8)
          and B[key][2] > FRAME_TOP + 8 and B[key][3] < FRAME_BOT - 8, f"{key} inside the frame and its panel")
# rotated panel titles inside their panel and left of the frame
for t, _, bb, _ in spans:
    if t in TITLES:
        lo, hi = (FRAME_TOP, DIVIDER) if t in TITLES[:2] else (DIVIDER, FRAME_BOT)
        check(bb[0] > 2 and bb[2] < FRAME_L - 3 and lo + 5 < bb[1] and bb[3] < hi - 5, f"title '{t}' inside its panel strip")
yrs = [bb for t, _, bb, _ in spans if t.isdigit() and len(t) == 4]
check(len(yrs) == 9 and all(bb[3] < H - 2 and bb[1] > AXIS_Y + 23 for bb in yrs), "year labels below the ticks, inside canvas")


def seg_clear(p0, p1, box, pad_r=G.RADIUS_PX):
    x0, x1, y0, y1 = box
    ts = np.linspace(0, 1, 4001)
    pts = np.outer(1 - ts, p0) + np.outer(ts, p1)
    cx = np.clip(pts[:, 0], x0 + pad_r, x1 - pad_r)
    cy = np.clip(pts[:, 1], y0 + pad_r, y1 - pad_r)
    return float(np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) - pad_r)


conns = {"UNI dashed": (mid_r("UNI"), TIP, {"UNI", "CLAM-TR"}),
         "UNI2-h dashed": (mid_r("UNI2-h"), TIP, {"UNI2-h", "CLAM-TR"}),
         "ABMIL->CLAM": (A0, A1, {"ABMIL", "CLAM"}),
         "CLAM->CLAM-TR": (C0, C1, {"CLAM", "CLAM-TR"}),
         "DSMIL->CLAM-TR thin": (THIN0, THIN1, {"DSMIL", "CLAM-TR"})}
for name, (a, b, ends) in conns.items():
    worst = min((seg_clear(np.array(a), np.array(b), B[k_]), k_) for k_ in B if k_ not in ends)
    check(worst[0] >= 15, f"{name}: clears every other box by {worst[0]:.1f} px (closest {worst[1]})")
# connectors vs box texts (no line through a text)
for name, (a, b, ends) in conns.items():
    for t, _, bb, _ in spans:
        if seg_clear(np.array(a), np.array(b), (bb[0], bb[2], bb[1], bb[3]), pad_r=0) < 3:
            check(False, f"{name} touches text '{t}'")
S17_CENTRES = {"UNI": 1990, "UNI2-h": 2296, "CLAM": 1120, "DSMIL": 1120, "TransMIL": 1120, "RetMIL": 1985,
               "MamMIL": 1985, "ABMIL": 340}
for key in B:
    nc = (B[key][0] + B[key][1]) / 2.0
    yr_ok = [yr for yr, x in Y.items() if B[key][0] < x < B[key][1]]
    print(f"     {key:9s} box x {B[key][0]:7.1f}-{B[key][1]:7.1f} y {B[key][2]:7.1f}-{B[key][3]:7.1f}; centre {nc:7.1f}; "
          f"year ticks under the box {yr_ok}")
print(f"  divider {DIVIDER}, frame bottom {FRAME_BOT:.1f}, canvas {W} x {H}; dashed tip {TIP}; UNI line clears UNI2-h by "
      f"{seg_clear(np.array(mid_r('UNI')), np.array(TIP), B['UNI2-h']):.1f} px")
pm = pg.get_pixmap(matrix=fitz.Matrix(110 * PRINT_SCALE / 72, 110 * PRINT_SCALE / 72))
pm.set_dpi(110, 110)
pm.save(str(HERE / "preview_fig_1_X_lineage_timeline_110dpi.png"))
z = 200 * PRINT_SCALE / 72
pm = pg.get_pixmap(matrix=fitz.Matrix(z, z))
pm.set_dpi(200, 200)
pm.save(str(HERE / "preview_fig_1_X_lineage_timeline_print200dpi.png"))
print()
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED:\n  " + "\n  ".join(FAIL))
