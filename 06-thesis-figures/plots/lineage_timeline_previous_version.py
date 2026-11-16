#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/lineage_timeline_previous_version.py
# Thesis output : previous version of Figure 2; make_fig2_lineage_timeline.py reads its box texts and line breaks as the reference content
# Inputs        : constants of lineage_timeline_base.py; the thesis build's PNG under CLAM_TR_PROJECT_DIR/thesis/main/figures
# Outputs       : out/ and previews/ next to this script
# Run           : read by make_fig2_lineage_timeline.py (can also be run on its own)
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Previous version of thesis Figure 2 (fig:lineage_timeline, figure file fig_1_X_lineage_timeline.png), drawn
with every text at >= 7 pt at its printed size. make_fig2_lineage_timeline.py reads its box texts and line breaks
(NEWLINES) as the reference content. Writes into the subfolders out/ and previews/ next to this script.

Source of the drawing: lineage_timeline_base.py, the base layout
(imported for its constants: box contents, colours, frame, axis, year ticks, line widths). Same canvas
(2752 x 1536 px at 200 dpi, printed at 0.95 \textwidth = 417.40 bp, i.e. 0.42131 bp per figure pt).

What changes relative to the base layout (nothing else):
  * font sizes: names 16.0 -> 18.5 pt, descriptors 14.2 (auto-fit down to 11.95) -> 17.0 pt, years 16.4 -> 17.0 pt,
    panel titles 16.6 -> 17.5 pt; printed: >= 7.16 pt (was 5.03-6.99 pt). No auto-fit shrinking (asserted).
  * boxes re-sized to the larger text; six descriptor lines re-broken so that the boxes keep their year columns
    (same words, same order): UNI / UNI2-h "1,0xx-d features" -> "1,0xx-d" + "features"; CONCH "Vision-language" ->
    "Vision-" + "language"; TransMIL "Nystrom Transformer" + "MIL" -> "Nystrom" + "Transformer MIL"; DTFD-MIL
    "Double-Tier Distillation" + "MIL" -> "Double-Tier" + "Distillation MIL"; RetMIL "Hierarchical Retention" +
    "MIL" -> "Hierarchical" + "Retention MIL".
  * box positions adjusted (2024 encoder stack now contiguous; UNI2-h next to it; dashed tip on the CLAM-TR top
    border) so that no box overlaps and every connector clears every box by >= 15 px (asserted).
Checks (all asserted): canvas; every PDF text span >= 7.0 pt printed; per-box text equal to the base layout (line
breaks aside); word multiset and number tokens of the PDF text equal to the base-layout PDF; each line inside its
box; boxes pairwise disjoint; connector clearances; box centres within 60 px of their base-layout centres.
Usage: python -P lineage_timeline_previous_version.py
"""
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

HERE = pathlib.Path(__file__).resolve().parent
OUT, PREV = HERE / "out", HERE / "previews"
OUT.mkdir(exist_ok=True)
PREV.mkdir(exist_ok=True)
ROOT = pathlib.Path(os.environ.get("CLAM_TR_PROJECT_DIR", pathlib.Path(__file__).resolve().parents[2]))  # repo path
GEN = HERE / "lineage_timeline_base.py"  # repo path
OLD_TWIN = GEN.parent / "staged" / "fig_1_X_lineage_timeline.pdf"
OLD_PNG = ROOT / "thesis/main/figures/fig_1_X_lineage_timeline.png"
spec = importlib.util.spec_from_file_location("G14", GEN)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)          # constants only; its main() is not run

W, H, DPI = G.W, G.H, G.DPI
PT = 72.0 / DPI
PRINT_SCALE = (0.95 * 441.01773 * 72 / 72.27) / (W / DPI * 72)          # 0.95 \textwidth (main.log) / figure width
MIN_PRINTED = 7.0
SIZE_NAME, SIZE_DESC, SIZE_TITLE, SIZE_YEAR = 18.5, 17.0, 17.5, 17.0
PITCH_PX = 49.0                     # base layout: 41 px for 14.2 pt descriptors; scaled by 17 / 14.2
PAD_H, PAD_V = 14.0, 17.0
GAP = 10.0                          # vertical gap inside a stack (base layout: ~10 px)
FONT = G.FONT

OLD = {b[0]: b for b in G.BOXES}
NEWLINES = {
    "UNI": ["UNI", "ViT-L/16", "1,024-d", "features"],
    "UNI2-h": ["UNI2-h", "ViT-H/14", "1,536-d", "features"],
    "Virchow": ["Virchow", "ViT-H/14"],
    "GigaPath": ["GigaPath", "ViT-g/14"],
    "CONCH": ["CONCH", "Vision-", "language"],
    "CLAM-TR": ["CLAM-TR", "Attention-Enhanced", "Dual-Branch"],
    "ABMIL": ["ABMIL", "Attention MIL", "Aggregator"],
    "CLAM": ["CLAM", "Encoder-Agnostic", "MIL"],
    "DSMIL": ["DSMIL", "Dual-Stream MIL"],
    "TransMIL": ["TransMIL", "Nyström", "Transformer MIL"],
    "DTFD-MIL": ["DTFD-MIL", "Double-Tier", "Distillation MIL"],
    "RetMIL": ["RetMIL", "Hierarchical", "Retention MIL"],
    "MamMIL": ["MamMIL", "State-Space MIL"],
}


def joined(lines):
    s = ""
    for ln in lines:
        s += ln if (not s or s.endswith("-") and not s.endswith(" -")) else " " + ln
    return s


# same words in the same order as the base layout (line breaks aside)
for key, lines in NEWLINES.items():
    old = OLD[key][6]
    if key == "CONCH":
        assert joined(lines) == "CONCH Vision-language" == " ".join(old), (key, joined(lines), old)
    else:
        assert " ".join(lines) == " ".join(old), (key, lines, old)
assert set(NEWLINES) == set(OLD)

# ---------------------------------------------------------------- text metrics (px at 200 dpi)
mpl.rcdefaults()
mpl.rcParams["lines.scale_dashes"] = False
mpl.rcParams["pdf.fonttype"] = 42
_f = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
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


# ---------------------------------------------------------------- layout
Y = G.YEAR_X
FRAME_TOP, DIVIDER, FRAME_BOT = 39.0, 767.5, 1279.0
W_ENC = max(need_w(k) for k in ("UNI", "UNI2-h", "Virchow", "GigaPath", "CONCH"))
W_21 = max(need_w(k) for k in ("CLAM", "DSMIL", "TransMIL"))
W_24 = max(need_w(k) for k in ("RetMIL", "MamMIL"))
C_STACK = Y[2024] + 13.0            # 2024 encoder stack (base layout: box centre 10 px right of the tick)
C_21 = 1120.0                       # 2021 aggregator stack (base layout: 1146); DTFD-MIL centre then = base 1487
C_24 = 1985.0                       # RetMIL / MamMIL column (base layout: 1986 / 1983)
B = {}


def put(key, xc, w, y0=None, y1=None):
    h = box_h(key)
    if y0 is None:
        y0 = y1 - h
    B[key] = [xc - w / 2.0, xc + w / 2.0, y0, y0 + h]


# encoders: UNI on top; Virchow, GigaPath, CONCH bottom-aligned above the divider; UNI2-h next to the stack
put("UNI", C_STACK, W_ENC, y0=54.0)
put("CONCH", C_STACK, W_ENC, y1=747.0)
put("GigaPath", C_STACK, W_ENC, y1=B["CONCH"][2] - GAP)
put("Virchow", C_STACK, W_ENC, y1=B["GigaPath"][2] - GAP)
put("UNI2-h", C_STACK + W_ENC + 12.0, W_ENC, y1=747.0)
# aggregators
w_cltr = need_w("CLAM-TR")
put("CLAM-TR", 2699.0 - w_cltr / 2.0, w_cltr, y0=793.0)         # right edge as in the base layout (2699)
put("ABMIL", 340.0, need_w("ABMIL"), y0=960.0)
put("CLAM", C_21, W_21, y0=784.0)
put("DSMIL", C_21, W_21, y0=960.0)
put("TransMIL", C_21, W_21, y0=B["DSMIL"][3] + GAP)
w_dt = need_w("DTFD-MIL")
put("DTFD-MIL", B["TransMIL"][1] + 11.0 + w_dt / 2.0, w_dt, y0=B["TransMIL"][2])
put("RetMIL", C_24, W_24, y0=960.0)
put("MamMIL", C_24, W_24, y0=B["RetMIL"][3] + GAP)


def mid_l(k):
    b = B[k]
    return (b[0], (b[2] + b[3]) / 2.0)


def mid_r(k):
    b = B[k]
    return (b[1], (b[2] + b[3]) / 2.0)


# dashed encoder -> CLAM-TR tip: on the CLAM-TR top border, far enough right that the UNI line clears UNI2-h by 25 px
TIP_Y = B["CLAM-TR"][2] - 2.0                                     # base layout: tip 791 on a box whose top is 793
p0 = np.array(mid_r("UNI"))
u2 = B["UNI2-h"]
f = (u2[2] - p0[1]) / (TIP_Y - p0[1])
tip_x_min = p0[0] + (u2[1] + 25.0 - p0[0]) / f
TIP = (float(np.ceil(max(tip_x_min, (B["CLAM-TR"][0] + B["CLAM-TR"][1]) / 2.0) + 5.0)), TIP_Y)
assert B["CLAM-TR"][0] + G.RADIUS_PX + 25 < TIP[0] < B["CLAM-TR"][1] - G.RADIUS_PX - 25, TIP

CL_Y = (B["CLAM"][2] + B["CLAM"][3]) / 2.0                         # CLAM -> CLAM-TR, horizontal
C0, C1 = (B["CLAM"][1], CL_Y), (B["CLAM-TR"][0], CL_Y)
HEAD2 = (C1[0] - 26.0, CL_Y)                                       # base layout: second head 26 px before the first
LINE_END = C1[0] - 42.0                                            # base layout: thick line ends 42 px before the tip
THIN0, THIN1 = mid_r("DSMIL"), (C1[0] - 46.0, CL_Y + 0.4)         # base layout: thin line ends 46 px before the tip
A0, A1 = mid_r("ABMIL"), mid_l("CLAM")


# ---------------------------------------------------------------- drawing helpers (from lineage_timeline_base.py)
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
    ax.add_patch(plt.Rectangle((168, FRAME_TOP), 2709 - 168, FRAME_BOT - FRAME_TOP, fill=False, edgecolor="black",
                               linewidth=3 * PT, zorder=1))
    ax.plot([168, 2709], [DIVIDER, DIVIDER], color="black", linewidth=3.5 * PT, zorder=1)
    tc, bc = (FRAME_TOP + DIVIDER) / 2.0, (DIVIDER + FRAME_BOT) / 2.0
    for text, xb, yc in (("Foundation-Model", 88, tc), ("Encoders (pathology)", 137, tc),
                         ("WSI Multiple Instance", 88, bc), ("Learning Aggregators", 137, bc)):
        ax.text(xb, yc, text, rotation=90, rotation_mode="anchor", ha="center", va="baseline",
                fontsize=SIZE_TITLE, family=FONT, zorder=4)
    line(ax, (43, 1317), (2680, 1317), 7)
    head(ax, (2708, 1317), (1, 0), 38, 38)
    for yr, x in Y.items():
        ax.plot([x, x], [1293, 1340], color="black", linewidth=3.5 * PT, zorder=2, solid_capstyle="butt")
        ax.text(x, 1399, str(yr), ha="center", va="baseline", fontsize=SIZE_YEAR, family=FONT, zorder=4)
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
        fill = OLD[key][5]
        ax.add_patch(FancyBboxPatch((x0 + lw / 2, y0 + lw / 2), (x1 - x0) - lw, (y1 - y0) - lw,
                                    boxstyle=f"round,pad=0,rounding_size={G.RADIUS_PX - lw / 2}",
                                    facecolor=fill, edgecolor="black", linewidth=lw * PT, zorder=3))
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


PNG = OUT / "fig_1_X_lineage_timeline.png"
PDF = OUT / "fig_1_X_lineage_timeline.pdf"
draw(PNG, PDF)

# ---------------------------------------------------------------- checks
FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


from PIL import Image  # noqa: E402

im = Image.open(PNG)
check(im.size == (W, H) == Image.open(OLD_PNG).size, f"PNG canvas {im.size} = thesis PNG {Image.open(OLD_PNG).size}")
check(abs(im.info.get("dpi", (0, 0))[0] - DPI) < 0.5, f"PNG dpi {im.info.get('dpi')}")
pg = fitz.open(PDF)[0]
k = DPI / 72.0
spans = [(s["text"].strip(), s["size"], [v * k for v in s["bbox"]]) for b in pg.get_text("dict")["blocks"]
         for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
mins = min(s[1] for s in spans)
check(mins * PRINT_SCALE >= MIN_PRINTED, f"smallest text {mins:.2f} pt -> {mins * PRINT_SCALE:.2f} pt printed (>= 7.0)")
old_pg = fitz.open(OLD_TWIN)[0]
old_txt, new_txt = old_pg.get_text(), pg.get_text()
ow = collections.Counter(old_txt.split())
nw_list = new_txt.split()
nw = collections.Counter(nw_list)
if nw["Vision-"] == 1 and nw["language"] == 1:                     # the one hyphen break
    nw.subtract({"Vision-": 1, "language": 1})
    nw["Vision-language"] += 1
    nw = +nw
check(ow == nw, f"word multiset equal to the base-layout PDF ({sum(ow.values())} words)"
      + ("" if ow == nw else f": old-only {dict(ow - nw)}, new-only {dict(nw - ow)}"))
NUM = re.compile(r"(?<![A-Za-z0-9])[+\-\u2212\u2013]?\d+(?:[.,]\d+)*")
on, nn = collections.Counter(NUM.findall(old_txt)), collections.Counter(NUM.findall(new_txt))
check(on == nn, f"printed number tokens identical ({sum(on.values())}): {sorted(on.items())}")
for key, (x0, x1, y0, y1) in B.items():
    lw = G.BORDER_THICK_PX if key == "CLAM-TR" else G.BORDER_PX
    for ln in NEWLINES[key]:
        hit = [bb for t, _, bb in spans if t == ln and x0 <= (bb[0] + bb[2]) / 2 <= x1 and y0 <= (bb[1] + bb[3]) / 2 <= y1]
        ok = len(hit) == 1 and hit[0][0] >= x0 + lw + 2 and hit[0][2] <= x1 - lw - 2 and hit[0][1] >= y0 + lw \
            and hit[0][3] <= y1 - lw
        if not ok:
            check(False, f"{key}: '{ln}' inside its box {hit}")
check(True, "every box line found once and inside its box (failures listed above, if any)")
keys = list(B)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = B[keys[i]], B[keys[j]]
        if not (a[1] + 8 <= b[0] or b[1] + 8 <= a[0] or a[3] + 8 <= b[2] or b[3] + 8 <= a[2]):
            check(False, f"boxes {keys[i]} and {keys[j]} overlap or are < 8 px apart")
check(True, "boxes pairwise separated by >= 8 px")
for key in B:
    check(168 + 10 < B[key][0] and B[key][1] < 2709 - 8 and (B[key][3] < DIVIDER - 8 or B[key][2] > DIVIDER + 8)
          and B[key][2] > FRAME_TOP + 8 and B[key][3] < FRAME_BOT - 8, f"{key} inside the frame and its panel")


def seg_clear(p0, p1, box, pad_r=G.RADIUS_PX):
    """Smallest distance (px) from segment p0-p1 to a rounded box (corners of radius pad_r)."""
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
    worst = min((seg_clear(np.array(a), np.array(b), B[k]), k) for k in B if k not in ends)
    check(worst[0] >= 15, f"{name}: clears every other box by {worst[0]:.1f} px (closest {worst[1]})")
for key in B:
    oc = (OLD[key][1] + OLD[key][2]) / 2.0
    nc = (B[key][0] + B[key][1]) / 2.0
    check(abs(nc - oc) <= 60, f"{key:9s} centre x {nc:7.1f} vs base {oc:7.1f} (shift {nc - oc:+6.1f} px; "
          f"box {B[key][1] - B[key][0]:.0f} x {B[key][3] - B[key][2]:.0f}, base {OLD[key][2] - OLD[key][1]:.0f} x "
          f"{OLD[key][4] - OLD[key][3]:.0f})")
print(f"  dashed tip {TIP}; UNI line clears UNI2-h by "
      f"{seg_clear(np.array(mid_r('UNI')), np.array(TIP), B['UNI2-h']):.1f} px")
# previews: the vector twin rendered at 200 dpi AT PRINTED SIZE (page scaled by PRINT_SCALE), and old-vs-new sheet
_z = 200 * PRINT_SCALE / 72.0
_pm = fitz.open(PDF)[0].get_pixmap(matrix=fitz.Matrix(_z, _z))
_pm.set_dpi(200, 200)
_pm.save(str(PREV / "fig_1_X_lineage_timeline_print200dpi.png"))
side = Image.new("RGB", (1376, 1546), "white")
side.paste(Image.open(OLD_PNG).convert("RGB").resize((1376, 768)), (0, 0))
side.paste(Image.open(PNG).convert("RGB").resize((1376, 768)), (0, 778))
side.save(PREV / "fig_1_X_lineage_timeline_old_top_new_bottom.png")
print()
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED:\n  " + "\n  ".join(FAIL))
