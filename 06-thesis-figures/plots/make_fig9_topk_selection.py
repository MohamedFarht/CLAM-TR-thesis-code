#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig9_topk_selection.py
# Thesis output : Figure 9 (fig_2_4_topk_selection)
# Inputs        : fold-0 attention dump of UNI2-h SpatialMultRetNet_bestHP (CLAM_TR_V20_DIR/UNI2-h/predictions_attention/; not in the repository) and the thumbnail of the previous figure file under CLAM_TR_PROJECT_DIR/thesis/main/figures
# Outputs       : PDF + PNG + preview next to this script
# Run           : python -P 06-thesis-figures/plots/make_fig9_topk_selection.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 9 (fig:topk_selection, figure file fig_2_4_topk_selection.pdf) re-set in Times New Roman, every
text item 10 pt printed (KTU thesis guide, section 5.1: figure text >= 10 pt). Writes
fig_2_4_topk_selection.{pdf,png} and a 110-dpi preview next to this script.

Inputs: the previous version of the figure file (right panel = the model's SAVED topk_indices; read from the figure
folder of the thesis build under CLAM_TR_PROJECT_DIR) for the two thumbnail rasters and the drawn dot
positions/colours; the saved fold-0 attention dump (read-only)
  CLAM_TR_V20_DIR/UNI2-h/predictions_attention/SpatialMultRetNet_bestHP_fold0_attention.json
for slide 48ae1bdf69446ba5fb0b7fe8fc2863c5 (attn_softmax, coords, topk_indices), exactly as the previous generator.
Drawing: the previous generator's draw() (imshow thumb; scatter(c=attn, cmap='viridis', s=4, alpha=0.7); colorbar with
ticks 0.0-0.8; scatter(saved topk, c='lime', s=18, alpha=0.9, edgecolors='black', linewidths=0.4)), but on a page
of the PRINTED size (old page x 0.48716, so font sizes are printed sizes; dot sizes scaled with the page), Times New
Roman 10 pt, and the panels moved down by the extra height the two 10-pt title lines need.
Asserted: the installed left dots = the dump's 331 patches (linear position fit, colours); the installed lime dots
= the dump's saved topk_indices in order (residual < 0.01 pt) and the new figure draws exactly those indices; words
and number tokens identical; smallest text >= 10 pt; dots at the same relative image positions with the same
colours and relative size; thumbnails pixel-identical; colour bar and tick positions unchanged relative to the bar.
Usage: python -P make_fig9_topk_selection.py
"""
import collections
import gc
import json
import os  # repo path (reads the environment variables of this repository)
import pathlib
import re
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

import fitz  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("CLAM_TR_PROJECT_DIR", pathlib.Path(__file__).resolve().parents[2]))  # repo path
OLD_PDF = ROOT / "thesis/main/figures/fig_2_4_topk_selection.pdf"
DUMP = pathlib.Path(os.environ.get("CLAM_TR_V20_DIR", HERE.parent / "data" / "v20_fullscale")) / "UNI2-h" / "predictions_attention" / "SpatialMultRetNet_bestHP_fold0_attention.json"  # repo path
SLIDE = "48ae1bdf69446ba5fb0b7fe8fc2863c5"
PRINT_W_BP = 0.95 * 441.01773 * 72 / 72.27
FS, FONT, K = 10.0, "Times New Roman", 128
TOP_NEW = 27.0                                   # printed bp from the page top to the panel tops (was 21.43)
FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


# ---------------------------------------------------------------- installed figure
old = fitz.open(OLD_PDF)
pg = old[0]
PW, PH = pg.rect.width, pg.rect.height
k = PRINT_W_BP / PW
rects, imgs = {}, {}
for im in pg.get_images(full=True):
    (r,) = pg.get_image_rects(im[0])
    if im[5] == "DeviceRGB":
        key = "left" if r.x0 < 300 else "right"
        pix = fitz.Pixmap(old, im[0])
        imgs[key] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[::-1].copy()  # stored bottom-up
        rects[key] = r
    else:
        rects["cbar"] = r
L, R, CB = rects["left"], rects["right"], rects["cbar"]
dr = pg.get_drawings()
is_lime = lambda d: d["fill"][1] > 0.9 and d["fill"][0] < 0.1 and d["fill"][2] < 0.1  # noqa: E731
left = [d for d in dr if d.get("fill") and d["rect"].width < 6 and L.contains(d["rect"]) and not is_lime(d)]
right = [d for d in dr if d.get("fill") and d["rect"].width < 12 and R.contains(d["rect"]) and is_lime(d)]
assert len(left) == 331 and len(right) == K, (len(left), len(right))
ctr = lambda ds: np.array([[(d["rect"].x0 + d["rect"].x1) / 2, (d["rect"].y0 + d["rect"].y1) / 2] for d in ds])  # noqa
lc, rc = ctr(left), ctr(right)
lcol = np.array([d["fill"] for d in left])
dl_old, dr_old = left[0]["rect"].width, right[0]["rect"].width          # marker diameters (pt, old page)
lw_l_old, lw_r_old = left[0]["width"], right[0]["width"]                  # marker edge widths (pt, old page)
ticks_old = {s["text"]: (s["bbox"][1] + s["bbox"][3]) / 2 for b in pg.get_text("dict")["blocks"]
             for l in b.get("lines", []) for s in l["spans"] if re.fullmatch(r"0\.\d", s["text"])}
old_text = pg.get_text()
print(f"installed page {PW:.2f} x {PH:.2f} pt (print scale {k:.5f}); marker diameters {dl_old:.3f} / {dr_old:.3f} pt")


def fit(xy_page, xy_data):
    out = []
    for j in (0, 1):
        A = np.c_[np.ones(len(xy_data)), xy_data[:, j]]
        coef, *_ = np.linalg.lstsq(A, xy_page[:, j], rcond=None)
        out.append((coef, float(np.abs(A @ coef - xy_page[:, j]).max())))
    return out


# ---------------------------------------------------------------- the saved dump
J = json.loads(DUMP.read_text(encoding="utf-8"))
meta = {kk: v for kk, v in J.items() if kk != "records"}
(rec,) = [r for r in J["records"] if r["slide_id"] == SLIDE]
del J
gc.collect()
attn, coords = np.array(rec["attn_softmax"], float), np.array(rec["coords"], float)
topk = np.array(rec["topk_indices"], int)
N = len(attn)
print(f"dump {meta.get('experiment')} fold {meta.get('fold')}; slide {SLIDE} label {rec['label']} pred {rec['pred']} N {N}")
(fx, ex), (fy, ey) = fit(lc, coords)
col = matplotlib.colormaps["viridis"](mcolors.Normalize(vmin=attn.min(), vmax=attn.max())(attn))[:, :3]
check(max(ex, ey) < 0.01 and float(np.abs(col - lcol).max()) < 1e-4,
      f"installed left dots = the dump's {N} patches (position residual {max(ex, ey):.1e} pt, colour |d| "
      f"{float(np.abs(col - lcol).max()):.1e})")
check(len(topk) == K and len(set(topk.tolist())) == K, f"saved topk_indices: {len(topk)} distinct patches")
(gx, rx), (gy, ry) = fit(rc, coords[topk])
check(max(rx, ry) < 0.01, f"installed lime dots = the saved topk_indices, in order (residual {max(rx, ry):.1e} pt)")
lpos = np.c_[fx[0] + fx[1] * coords[:, 0], fy[0] + fy[1] * coords[:, 1]]
rpos = np.c_[gx[0] + gx[1] * coords[topk, 0], gy[0] + gy[1] * coords[topk, 1]]

# ---------------------------------------------------------------- new page (printed bp, y down)
DY = TOP_NEW - L.y0 * k
H = PH * k + DY
sc_r = lambda r: fitz.Rect(r.x0 * k, r.y0 * k + DY, r.x1 * k, r.y1 * k + DY)  # noqa: E731
nL, nCB, nR = sc_r(L), sc_r(CB), sc_r(R)
WP = PRINT_W_BP
print(f"new page {WP:.2f} x {H:.2f} bp; left {nL}; bar {nCB}; right {nR}")


def axes_at(fig, r):
    return fig.add_axes([r.x0 / WP, 1 - r.y1 / H, r.width / WP, r.height / H])


plt.rcdefaults()
plt.rcParams.update({"pdf.fonttype": 42, "font.family": FONT, "mathtext.fontset": "stix", "font.size": FS,
                     "text.color": ".15", "ytick.color": ".15", "axes.edgecolor": ".8", "axes.labelcolor": ".15",
                     "ytick.major.size": 3.5 * k, "ytick.major.width": 0.8 * k, "ytick.major.pad": 3.5 * k})
fig = plt.figure(figsize=(WP / 72, H / 72), dpi=600)
ax0 = axes_at(fig, nL)
ax0.imshow(imgs["left"], extent=(L.x0, L.x1, L.y1, L.y0), interpolation="none")
sc = ax0.scatter(lpos[:, 0], lpos[:, 1], c=attn, cmap="viridis", s=(dl_old * k) ** 2, alpha=0.7, linewidths=lw_l_old * k)
ax0.set_xlim(L.x0, L.x1)
ax0.set_ylim(L.y1, L.y0)
ax0.axis("off")
cb = fig.colorbar(sc, cax=axes_at(fig, nCB), ticks=[0.0, 0.2, 0.4, 0.6, 0.8])
cb.ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8"], fontsize=FS)
cb.outline.set_linewidth(0.8 * k)
ax1 = axes_at(fig, nR)
ax1.imshow(imgs["right"], extent=(R.x0, R.x1, R.y1, R.y0), interpolation="none")
ax1.scatter(rpos[:, 0], rpos[:, 1], c="lime", s=(dr_old * k) ** 2, alpha=0.9, edgecolors="black",
            linewidths=lw_r_old * k)
ax1.set_xlim(R.x0, R.x1)
ax1.set_ylim(R.y1, R.y0)
ax1.axis("off")
ty = 1 - 1.5 / H
fig.text((nL.x0 + nL.x1) / 2 / WP, ty, f"(a) All extracted patches\n(N={N}, color=attention weight)",
         ha="center", va="top", fontsize=FS, linespacing=1.1)
fig.text((nR.x0 + nR.x1) / 2 / WP, ty, f"(b) TopK={K} selected for the\ndual-branch aggregator", ha="center",
         va="top", fontsize=FS, linespacing=1.1)
NEW_PDF, NEW_PNG = HERE / "fig_2_4_topk_selection.pdf", HERE / "fig_2_4_topk_selection.png"
fig.savefig(NEW_PDF, metadata={"CreationDate": None, "ModDate": None})
fig.savefig(NEW_PNG, dpi=300 / k)            # same pixel count per printed inch as the installed 300-dpi PNG
plt.close(fig)

# ---------------------------------------------------------------- checks on the staged figure
new = fitz.open(NEW_PDF)
np_ = new[0]
check(abs(np_.rect.width - PRINT_W_BP) < 0.05, f"page width {np_.rect.width:.2f} bp = printed width (scale 1)")
spans = [(s["text"], s["size"], fitz.Rect(s["bbox"]), s["font"]) for b in np_.get_text("dict")["blocks"]
         for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
check(min(s[1] for s in spans) >= 9.95, f"smallest text {min(s[1] for s in spans):.2f} pt printed (installed "
      f"{min(s['size'] for b in pg.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans'] if s['text'].strip()) * k:.2f})")
check({s[3] for s in spans} == {"TimesNewRomanPSMT"}, f"fonts {sorted({s[3] for s in spans})}")
new_text = np_.get_text()
ow, nw = collections.Counter(old_text.split()), collections.Counter(new_text.split())
check(ow == nw, f"words identical ({sum(ow.values())})" + ("" if ow == nw else f" old-only {dict(ow - nw)} new-only {dict(nw - ow)}"))
NUM = re.compile(r"(?<![A-Za-z0-9])[+\-\u2212\u2013]?\d+(?:[.,]\d+)*")
on, nn = collections.Counter(NUM.findall(old_text)), collections.Counter(NUM.findall(new_text))
check(on == nn, f"printed number tokens identical: {sorted(on.items())}")
ndr = np_.get_drawings()
im_rects, im_arr, cbar_new = {}, {}, None
for im in np_.get_images(full=True):
    r = np_.get_image_rects(im[0])[0]
    if im[5] == "DeviceRGB":
        key = "left" if r.x0 < 150 else "right"
        im_rects[key] = r
        pix = fitz.Pixmap(new, im[0])
        im_arr[key] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3][::-1]
    else:
        cbar_new = r
for key in ("left", "right"):
    check(im_arr[key].shape == imgs[key].shape and np.array_equal(im_arr[key], imgs[key]),
          f"{key} thumbnail pixel-identical (same stored raster)")
nl = [d for d in ndr if d.get("fill") and d["rect"].width < 3 and im_rects["left"].contains(d["rect"]) and not is_lime(d)]
nr = [d for d in ndr if d.get("fill") and d["rect"].width < 6 and im_rects["right"].contains(d["rect"]) and is_lime(d)]
check(len(nl) == N and len(nr) == K, f"dots: {len(nl)} viridis + {len(nr)} lime")
rel = lambda c, r: np.c_[(c[:, 0] - r.x0) / r.width, (c[:, 1] - r.y0) / r.height]  # noqa: E731
d1 = float(np.abs(rel(ctr(nl), im_rects["left"]) - rel(lc, L)).max())
check(d1 < 2e-4, f"viridis dots at the same relative image positions (max |d| {d1:.1e} of the panel)")
check(float(np.abs(np.array([d["fill"] for d in nl]) - lcol).max()) < 1e-4, "viridis dot colours identical")
d2 = float(np.abs(rel(ctr(nr), im_rects["right"]) - rel(rc, R)).max())
check(d2 < 2e-4, f"lime dots at the same relative image positions, same order = saved topk (max |d| {d2:.1e})")
check(abs(nl[0]["rect"].width / im_rects["left"].width - dl_old / L.width) < 1e-3
      and abs(nr[0]["rect"].width / im_rects["right"].width - dr_old / R.width) < 1e-3, "dot sizes relative to the panels unchanged")
check(cbar_new is not None and abs(cbar_new.height / im_rects["left"].height - CB.height / L.height) < 2e-3,
      "colour bar spans the same relative height")
ticks_new = {s[0]: (s[2].y0 + s[2].y1) / 2 for s in spans if re.fullmatch(r"0\.\d", s[0])}
dt = max(abs((ticks_new[t] - cbar_new.y0) / cbar_new.height - (ticks_old[t] - CB.y0) / CB.height) for t in ticks_old)
check(set(ticks_new) == set(ticks_old) and dt < 3e-3, f"tick labels at the same relative bar positions (max |d| {dt:.1e})")
# colour bar rendering
za, zb = 4.0, 4.0 * CB.width / cbar_new.width
A = old[0].get_pixmap(clip=CB, matrix=fitz.Matrix(za, za))
B = np_.get_pixmap(clip=cbar_new, matrix=fitz.Matrix(zb, zb))
from PIL import Image  # noqa: E402
Aa = np.frombuffer(A.samples, dtype=np.uint8).reshape(A.height, A.width, A.n)[..., :3].astype(float)
Ba = np.asarray(Image.fromarray(np.frombuffer(B.samples, dtype=np.uint8).reshape(B.height, B.width, B.n)[..., :3])
                .resize((A.width, A.height), Image.LANCZOS), dtype=float)
inner = (slice(6, -6), slice(6, -6))
check(float(np.abs(Aa[inner] - Ba[inner]).mean()) < 3.0, f"colour bar rendered like the installed one (mean |dRGB| "
      f"{float(np.abs(Aa[inner] - Ba[inner]).mean()):.2f} of 255)")
for key, ro in (("left", L), ("right", R)):                 # rendered panels (thumbnail + dots), orientation
    rn = im_rects[key]
    a = old[0].get_pixmap(clip=ro, matrix=fitz.Matrix(1.5, 1.5))
    zz = 1.5 * ro.width / rn.width
    b = np_.get_pixmap(clip=rn, matrix=fitz.Matrix(zz, zz))
    Aa2 = np.frombuffer(a.samples, dtype=np.uint8).reshape(a.height, a.width, a.n)[..., :3].mean(2)
    Bb2 = np.asarray(Image.fromarray(np.frombuffer(b.samples, dtype=np.uint8).reshape(b.height, b.width, b.n)[..., :3]
                                     .mean(2).astype(np.uint8)).resize((a.width, a.height), Image.LANCZOS), dtype=float)
    c, cf = np.corrcoef(Aa2.ravel(), Bb2.ravel())[0, 1], np.corrcoef(Aa2.ravel(), Bb2[::-1].ravel())[0, 1]
    check(c > 0.97 and c > cf + 0.3, f"{key} panel rendered like the installed one, same orientation (r = {c:.4f}; "
          f"upside-down {cf:.4f})")
outside = [t for t, _, r, _ in spans if r.x0 < 0 or r.x1 > np_.rect.width or r.y0 < -0.5 or r.y1 > np_.rect.height]
check(not outside, f"all text inside the page {outside}")
panel_clash = [t for t, _, r, _ in spans for pr in (im_rects["left"], im_rects["right"], cbar_new)
               if (r & pr).height > 0.3 and (r & pr).width > 0.3]
check(not panel_clash, f"no text overlaps an image or the colour bar {panel_clash}")
pm = np_.get_pixmap(dpi=110)
pm.save(str(HERE / "preview_fig_2_4_topk_selection_110dpi.png"))
print(f"printed size {np_.rect.width / 72 * 2.54:.2f} x {np_.rect.height / 72 * 2.54:.2f} cm "
      f"(installed {PW * k / 72 * 2.54:.2f} x {PH * k / 72 * 2.54:.2f} cm)")
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED:\n  " + "\n  ".join(FAIL))
sys.exit(1 if FAIL else 0)
