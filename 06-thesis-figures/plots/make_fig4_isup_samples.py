#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig4_isup_samples.py
# Thesis output : Figure 4 (fig_2_3_isup_samples), re-set from the previous version of the figure
# Inputs        : the thumbnails and texts of the previous figure file under CLAM_TR_PROJECT_DIR/thesis/main/figures; representative_slides.json in the run's v20 results folder (CLAM_TR_V20_DIR/UNI2-h; not in the repository)
# Outputs       : PDF + PNG + preview next to this script
# Run           : python -P 06-thesis-figures/plots/make_fig4_isup_samples.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 4 (fig:isup_samples, figure file fig_2_3_isup_samples.pdf) re-set in Times New Roman with every
text item at 10 pt printed (KTU thesis guide, section 5.1: figure text >= 10 pt). Writes
fig_2_3_isup_samples.{pdf,png} and a 110-dpi preview next to this script.

Input: the previous version of the figure file (fig_2_3_isup_samples.pdf in the figure folder of the thesis build
under CLAM_TR_PROJECT_DIR): its six embedded thumbnails (pixel data extracted unchanged with PyMuPDF), its grade
labels, slide-ID prefixes and title (text unchanged). The slide IDs are re-checked against
CLAM_TR_V20_DIR/UNI2-h/representative_slides.json (results folder CLAM_TR_Results/v20_fullscale; read-only).
Layout as in the previous version (drawn at the printed width 0.95 \textwidth, so font sizes are printed sizes); the
title blocks are taller for the 10 pt text and the figure height grows by that amount, so every thumbnail keeps its
printed size.
Asserted: thumbnails pixel-identical, words and number tokens identical, titles of a row on one baseline, smallest
text >= 10 pt, titles clear of the images, all text inside the page, only Times New Roman.
Usage: python -P make_fig4_isup_samples.py [--no-drive]
"""
import collections
import hashlib
import json
import os  # repo path (reads the environment variables of this repository)
import pathlib
import re
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fitz  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("CLAM_TR_PROJECT_DIR", pathlib.Path(__file__).resolve().parents[2]))  # repo path
OLD_PDF = ROOT / "thesis/main/figures/fig_2_3_isup_samples.pdf"
REP_JSON = pathlib.Path(os.environ.get("CLAM_TR_V20_DIR", HERE.parent / "data" / "v20_fullscale")) / "UNI2-h" / "representative_slides.json"  # repo path
PRINT_W_BP = 0.95 * 441.01773 * 72 / 72.27                # 0.95 \textwidth
FS = 10.0
FONT = "Times New Roman"
FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


# ---------------------------------------------------------------- read the installed figure
old = fitz.open(OLD_PDF)
pg = old[0]
spans = [(s["text"], s["size"], fitz.Rect(s["bbox"])) for b in pg.get_text("dict")["blocks"]
         for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
labels = [s for s in spans if re.match(r"G[0-5] ", s[0])]
ids = [s for s in spans if re.fullmatch(r"[0-9a-f]{12}\.\.\.", s[0])]
sup = [s for s in spans if s[0].startswith("Example whole slide image")]
assert len(labels) == 6 and len(ids) == 6 and len(sup) == 1, (labels, ids, sup)
SUP = sup[0][0]
panels = []
for img in pg.get_images(full=True):
    xref = img[0]
    (r,) = pg.get_image_rects(xref)
    pix = fitz.Pixmap(old, xref)
    assert pix.n == 3 and pix.alpha == 0, (xref, pix.n, pix.alpha)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[::-1].copy()  # stored bottom-up
    cx = (r.x0 + r.x1) / 2
    near = lambda s: abs((s[2].x0 + s[2].x1) / 2 - cx) + (1e6 if s[2].y1 > r.y0 else abs(r.y0 - s[2].y1))  # noqa: E731
    lab, sid = min(labels, key=near), min(ids, key=near)
    panels.append({"grade": int(lab[0][1]), "label": lab[0], "id": sid[0], "img": arr, "old_rect": r})
panels.sort(key=lambda p: p["grade"])
assert [p["grade"] for p in panels] == list(range(6))
print("installed figure:", [(p["grade"], p["label"], p["id"], p["img"].shape[1::-1]) for p in panels])
old_w = {p["grade"]: (p["old_rect"].width, p["old_rect"].height) for p in panels}

if "--no-drive" not in sys.argv:
    rep = json.loads(REP_JSON.read_text(encoding="utf-8"))
    assert rep["fold"] == 0 and rep["encoder"] == "UNI2-h", rep
    for p in panels:
        assert rep["representatives"][str(p["grade"])][:12] == p["id"][:12], (p["grade"], p["id"])
    print("  slide IDs of all six panels = representative_slides.json (UNI2-h, fold 0)")

# ---------------------------------------------------------------- layout (inches, printed size)
FW = PRINT_W_BP / 72.0
M, CG = 0.03, 0.08                                         # as in the previous version
CW = (FW - 2 * M - 2 * CG) / 3
IH = (591.12 / 832.78 * FW - 0.19 - 2 * (0.27 + 0.025) - 0.07 - 0.02) / 2   # previous cell height (images keep size)
T_SUP, T_TIT, G_TI, G_ROW, M_BOT, TOPM = 0.215, 0.335, 0.03, 0.07, 0.02, 0.025
FH = T_SUP + 2 * (T_TIT + G_TI + IH) + G_ROW + M_BOT
print(f"new figure {FW:.3f} x {FH:.3f} in ({FW * 72:.2f} x {FH * 72:.2f} bp); cell {CW:.3f} x {IH:.3f} in")

plt.rcdefaults()
plt.rcParams.update({"pdf.fonttype": 42, "font.family": FONT, "mathtext.fontset": "stix"})
fig = plt.figure(figsize=(FW, FH))
fig.text(0.5, 1 - TOPM / FH, SUP, ha="center", va="top", fontsize=FS)
for p in panels:
    g = p["grade"]
    row, col = divmod(g, 3)
    top = FH - T_SUP - row * (T_TIT + G_TI + IH + G_ROW)
    x0 = M + col * (CW + CG)
    fig.text((x0 + CW / 2) / FW, top / FH, f"{p['label']}\n{p['id']}", ha="center", va="top", fontsize=FS,
             linespacing=1.15)
    ih, iw = p["img"].shape[:2]
    sc = min(CW / iw, IH / ih)
    w, h = iw * sc, ih * sc
    ax = fig.add_axes([(x0 + (CW - w) / 2) / FW, (top - T_TIT - G_TI - h) / FH, w / FW, h / FH])
    ax.imshow(p["img"], interpolation="none", aspect="auto", origin="upper")
    ax.axis("off")
NEW_PDF, NEW_PNG = HERE / "fig_2_3_isup_samples.pdf", HERE / "fig_2_3_isup_samples.png"
fig.savefig(NEW_PDF, metadata={"CreationDate": None, "ModDate": None})
fig.savefig(NEW_PNG, dpi=600)
plt.close(fig)

# ---------------------------------------------------------------- checks
new = fitz.open(NEW_PDF)
npg = new[0]
check(abs(npg.rect.width - PRINT_W_BP) < 0.05, f"page width {npg.rect.width:.2f} bp = printed width (scale 1)")
nspans = [(s["text"], s["size"], fitz.Rect(s["bbox"]), s["font"]) for b in npg.get_text("dict")["blocks"]
          for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
check(min(s[1] for s in nspans) >= 9.95, f"smallest text {min(s[1] for s in nspans):.2f} pt printed")
check({s[3] for s in nspans} == {"TimesNewRomanPSMT"}, f"fonts {sorted({s[3] for s in nspans})}")
ow, nw = collections.Counter(pg.get_text().split()), collections.Counter(npg.get_text().split())
check(ow == nw, f"words identical ({sum(ow.values())})" + ("" if ow == nw else f" old-only {dict(ow - nw)} new-only {dict(nw - ow)}"))
NUM = re.compile(r"(?<![A-Za-z0-9])[+\-\u2212\u2013]?\d+(?:[.,]\d+)*")
on, nn = collections.Counter(NUM.findall(pg.get_text())), collections.Counter(NUM.findall(npg.get_text()))
check(on == nn, f"number tokens identical ({sum(on.values())})")
rows = collections.defaultdict(list)
for t, size, r, _ in nspans:
    if re.match(r"G[0-5] ", t):
        rows[int(t[1]) // 3].append(round(r.y0, 3))
check(all(len(set(v)) == 1 and len(v) == 3 for v in rows.values()), f"grade titles share one top per row {dict(rows)}")
imgs = npg.get_images(full=True)
new_h, new_rects = [], []
for im in imgs:
    pix = fitz.Pixmap(new, im[0])
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3][::-1]
    new_h.append(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    new_rects.append(npg.get_image_rects(im[0])[0])
old_h = sorted(hashlib.sha256(p["img"].tobytes()).hexdigest() for p in panels)
check(len(imgs) == 6 and sorted(new_h) == old_h, "6 thumbnails pixel-identical to the installed figure")
oldsz = sorted((round(p["old_rect"].width, 1), round(p["old_rect"].height, 1)) for p in panels)
newsz = sorted((round(r.width, 1), round(r.height, 1)) for r in new_rects)
check(oldsz == newsz, f"thumbnail printed sizes unchanged {newsz}")
clash = [(t, i) for t, _, r, _ in nspans for i, ir in enumerate(new_rects) if (r & ir).height > 0.3 and (r & ir).width > 0.3]
check(not clash, f"no text overlaps a thumbnail {clash}")
outside = [t for t, _, r, _ in nspans if r.x0 < 0 or r.x1 > npg.rect.width or r.y0 < -0.5 or r.y1 > npg.rect.height]
check(not outside, f"all text inside the page {outside}")
ov = [(a[0], b[0]) for i, a in enumerate(nspans) for b in nspans[i + 1:] if (a[2] & b[2]).height > 1.0 and (a[2] & b[2]).width > 0.3]
check(not ov, f"no text overlaps other text {ov}")
pm = npg.get_pixmap(dpi=110)
pm.save(str(HERE / "preview_fig_2_3_isup_samples_110dpi.png"))
print(f"printed size {npg.rect.width / 72 * 2.54:.2f} x {npg.rect.height / 72 * 2.54:.2f} cm "
      f"(installed {pg.rect.width / 72 * 2.54:.2f} x {pg.rect.height / 72 * 2.54:.2f} cm)")
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED:\n  " + "\n  ".join(FAIL))
sys.exit(1 if FAIL else 0)
