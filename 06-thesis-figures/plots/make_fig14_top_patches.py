#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig14_top_patches.py
# Thesis output : Figure 14 a-c (fig_3_X_top_patches_g1, _g3, _g5), re-set from the previous version of the figure files
# Inputs        : the patch rasters and texts of the previous figure files under CLAM_TR_PROJECT_DIR/thesis/main/figures
# Outputs       : PDF + PNG + previews next to this script
# Run           : python -P 06-thesis-figures/plots/make_fig14_top_patches.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 14 a-c (fig:top_patches_g1/g3/g5) re-set in Times New Roman, every text item 10 pt printed (KTU
thesis guide, section 5.1: figure text >= 10 pt). Writes fig_3_X_top_patches_g{1,3,5}.{pdf,png} and 110-dpi
previews next to this script.

Why the subfigure width changes: at 10 pt Times New Roman the label "attn=0.9987" is 47.9 bp wide, but a 3-column
grid inside the previous 0.31\textwidth subfigure (136.2 bp) leaves at most 45.4 bp per column even with no gaps.
Splitting the label token or abbreviating it is not allowed, so each file is drawn for a subfigure of
SUB_FRAC = 0.48\textwidth (210.9 bp): G1 and G3 side by side, G5 centred on a second row (same three files, same
subfigure captions in the thesis).

Inputs: the previous version of the three figure files fig_3_X_top_patches_g{1,3,5}.pdf (read from the figure
folder of the thesis build under CLAM_TR_PROJECT_DIR): the nine patch rasters of each grid (extracted unchanged
with PyMuPDF), the label of each patch ("Rank k:" / "attn=..."), and the four title lines. Layout: same 3 x 3 grid
in rank order (row-major), same printed patch size (39.6 bp), label in two lines above each patch, the title on as
few lines as fit (the previous line breaks kept where a joined title does not fit). Drawn at printed size, so font
sizes are printed sizes.
Asserted: words and number tokens identical; every label above its own patch; patches pixel-identical and in the
same rank order; rendered orientation unchanged; smallest text >= 10 pt; only Times New Roman; no text overlap;
all text inside the page; the three grids the same height.
Usage: python -P make_fig14_top_patches.py
"""
import collections
import os  # repo path (reads the environment variables of this repository)
import pathlib
import re
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fitz  # noqa: E402
from PIL import Image  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("CLAM_TR_PROJECT_DIR", pathlib.Path(__file__).resolve().parents[2]))  # repo path
FIG = ROOT / "thesis" / "main" / "figures"
SUB_FRAC = 0.48
W_BP = SUB_FRAC * 441.01773 * 72 / 72.27          # \linewidth inside the 0.48\textwidth subfigure
OLD_W_BP = 0.31 * 441.01773 * 72 / 72.27          # installed subfigure width (installed files are drawn at it)
FONT, FS = "Times New Roman", 10.0
PITCH = 1.15 * FS                                  # line pitch of titles and labels
COL_W, GAP_C, GAP_R, M, M_TOP = 50.0, 6.0, 3.0, 1.0, 2.0
FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def lines_of(pg):
    return [(" ".join(x["text"] for x in l["spans"]).strip(), fitz.Rect(l["bbox"]), max(x["size"] for x in l["spans"]),
             {x["font"] for x in l["spans"]})
            for b in pg.get_text("dict")["blocks"] for l in b.get("lines", []) if "".join(x["text"] for x in l["spans"]).strip()]


def read_old(path):
    d = fitz.open(path)
    pg = d[0]
    assert abs(pg.rect.width - OLD_W_BP) < 0.05, pg.rect
    ls = lines_of(pg)
    titles = sorted([t for t in ls if not re.fullmatch(r"Rank \d:|attn=\d\.\d{4}", t[0])], key=lambda t: t[1].y0)
    ranks = [t for t in ls if re.fullmatch(r"Rank \d:", t[0])]
    attns = [t for t in ls if re.fullmatch(r"attn=\d\.\d{4}", t[0])]
    assert len(titles) == 4 and len(ranks) == 9 and len(attns) == 9, (titles, len(ranks), len(attns))
    patches = []
    for im in pg.get_images(full=True):
        (r,) = pg.get_image_rects(im[0])
        pix = fitz.Pixmap(d, im[0])
        assert pix.n == 3 and not pix.alpha
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[::-1].copy()  # bottom-up
        cx = (r.x0 + r.x1) / 2
        a = [t for t in attns if abs((t[1].x0 + t[1].x1) / 2 - cx) < 2 and 0 <= r.y0 - t[1].y1 < 6]
        k = [t for t in ranks if abs((t[1].x0 + t[1].x1) / 2 - cx) < 2 and 0 <= a[0][1].y0 - t[1].y1 + 1.5 < 6] if a else []
        assert len(a) == 1 and len(k) == 1, (r, a, k)
        patches.append({"rect": r, "img": arr, "rank": k[0][0], "attn": a[0][0]})
    patches.sort(key=lambda p: (round(p["rect"].y0), p["rect"].x0))
    assert [int(p["rank"].split()[1].rstrip(":")) for p in patches] == list(range(1, 10))
    return d, pg, [t[0] for t in titles], patches


def width_bp(s, rend, fig):
    t = fig.text(0, 0, s, fontsize=FS, family=FONT)
    w = t.get_window_extent(rend).width * 72 / fig.dpi
    t.remove()
    return w


def build(g):
    name = f"fig_3_X_top_patches_g{g}"
    old_doc, old_pg, tl_old, patches = read_old(FIG / f"{name}.pdf")
    p_bp = patches[0]["rect"].width                                # installed printed patch size (scale 1), kept
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": FONT, "mathtext.fontset": "stix"})
    probe = plt.figure(figsize=(W_BP / 72, 2), dpi=72)
    rend = probe.canvas.get_renderer()
    avail = W_BP - 2 * M - 4.0
    tl = []
    for pair in (tl_old[0:2], tl_old[2:4]):                        # two logical titles, installed line breaks
        joined = " ".join(pair)
        tl += [joined] if width_bp(joined, rend, probe) <= avail else list(pair)
    assert all(width_bp(t, rend, probe) <= avail for t in tl), tl
    lab_w = max(width_bp(s, rend, probe) for p in patches for s in (p["rank"], p["attn"]))
    plt.close(probe)
    assert lab_w <= COL_W and p_bp <= COL_W, (lab_w, p_bp)
    grid_w = 3 * COL_W + 2 * GAP_C
    gx0 = (W_BP - grid_w) / 2
    lab_h = 2 * PITCH
    row_h = lab_h + 1.0 + p_bp
    H = M_TOP + len(tl) * PITCH + 3.0 + 3 * row_h + 2 * GAP_R + M
    fig = plt.figure(figsize=(W_BP / 72, H / 72))
    y = H - M_TOP
    for ln in tl:
        fig.text(0.5, y / H, ln, ha="center", va="top", fontsize=FS, family=FONT)
        y -= PITCH
    y_grid = H - M_TOP - len(tl) * PITCH - 3.0
    for i, p in enumerate(patches):
        r, c = divmod(i, 3)
        xc = gx0 + c * (COL_W + GAP_C) + COL_W / 2
        top = y_grid - r * (row_h + GAP_R)
        for j, s in enumerate((p["rank"], p["attn"])):
            fig.text(xc / W_BP, (top - j * PITCH) / H, s, ha="center", va="top", fontsize=FS, family=FONT)
        ax = fig.add_axes([(xc - p_bp / 2) / W_BP, (top - lab_h - 1.0 - p_bp) / H, p_bp / W_BP, p_bp / H])
        ax.imshow(p["img"], interpolation="none", origin="upper")
        ax.axis("off")
    pdf, png = HERE / f"{name}.pdf", HERE / f"{name}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=600)
    plt.close(fig)

    new = fitz.open(pdf)
    npg = new[0]
    check(abs(npg.rect.width - W_BP) < 0.05, f"G{g}: page width {npg.rect.width:.2f} bp = printed width (scale 1)")
    print(f"       printed {npg.rect.width / 72 * 2.54:.2f} x {npg.rect.height / 72 * 2.54:.2f} cm (installed "
          f"{old_pg.rect.width / 72 * 2.54:.2f} x {old_pg.rect.height / 72 * 2.54:.2f} cm); title lines {tl}")
    ot, nt = old_pg.get_text(), npg.get_text()
    ow, nw = collections.Counter(ot.split()), collections.Counter(nt.split())
    check(ow == nw, f"G{g}: words identical ({sum(ow.values())})" + ("" if ow == nw else f" old-only {dict(ow - nw)} new-only {dict(nw - ow)}"))
    NUM = re.compile(r"(?<![A-Za-z0-9])[+\-\u2212\u2013]?\d+(?:[.,]\d+)*")
    on, nn = collections.Counter(NUM.findall(ot)), collections.Counter(NUM.findall(nt))
    check(on == nn, f"G{g}: number tokens identical ({sum(on.values())})")
    lines = lines_of(npg)
    check(min(s for _, _, s, _ in lines) >= 9.95, f"G{g}: smallest text {min(s for _, _, s, _ in lines):.2f} pt printed")
    fonts = set().union(*[f for _, _, _, f in lines])
    check(fonts == {"TimesNewRomanPSMT"}, f"G{g}: fonts {sorted(fonts)}")
    out = [t for t, rb, _, _ in lines if rb.x0 < -0.2 or rb.x1 > npg.rect.width + 0.2 or rb.y0 < -0.2 or rb.y1 > npg.rect.height + 0.2]
    check(not out, f"G{g}: all text inside the page {out}")
    ov = [(a[0], b[0]) for i, a in enumerate(lines) for b in lines[i + 1:]
          if (a[1] & b[1]).width > 0.2 and (a[1] & b[1]).height > 0.08 * min(a[2], b[2])]
    check(not ov, f"G{g}: no text line overlaps another ({len(lines)} lines) {ov}")
    imr = []
    for im in npg.get_images(full=True):
        (r,) = npg.get_image_rects(im[0])
        pix = fitz.Pixmap(new, im[0])
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3][::-1]
        imr.append((r, a))
    imr.sort(key=lambda t: (round(t[0].y0), t[0].x0))
    check(len(imr) == 9 and all(np.array_equal(a, p["img"]) for (r, a), p in zip(imr, patches)),
          f"G{g}: 9 patches pixel-identical (same stored rasters), rank order row-major")
    clash = [t for t, rb, _, _ in lines for r, _ in imr if (rb & r).width > 0.3 and (rb & r).height > 0.3]
    check(not clash, f"G{g}: no text overlaps a patch {clash}")
    okpos = True
    for (r, _), p in zip(imr, patches):
        cx = (r.x0 + r.x1) / 2
        ha = [rb for t, rb, _, _ in lines if t == p["attn"] and abs((rb.x0 + rb.x1) / 2 - cx) < 1 and 0 <= r.y0 - rb.y1 < 4]
        hk = [rb for t, rb, _, _ in lines if t == p["rank"] and abs((rb.x0 + rb.x1) / 2 - cx) < 1 and ha
              and 0 <= ha[0].y0 - rb.y1 + 1.5 < 6]
        okpos &= len(ha) == 1 and len(hk) == 1
    check(okpos, f"G{g}: every 'Rank k:' / 'attn=' pair directly above its own patch")
    check(all(abs(r.width - p_bp) < 0.05 for r, _ in imr), f"G{g}: patch printed size {imr[0][0].width:.2f} bp = installed {p_bp:.2f} bp")
    cs = []
    for (rn, _), p in zip(imr, patches):                           # rendered orientation vs the installed file
        A = old_pg.get_pixmap(clip=p["rect"], matrix=fitz.Matrix(3, 3))
        B = npg.get_pixmap(clip=rn, matrix=fitz.Matrix(3, 3))
        Aa = np.frombuffer(A.samples, dtype=np.uint8).reshape(A.height, A.width, A.n)[..., :3].mean(2)
        Bb = np.asarray(Image.fromarray(np.frombuffer(B.samples, dtype=np.uint8).reshape(B.height, B.width, B.n)[..., :3]
                                        .mean(2).astype(np.uint8)).resize((A.width, A.height), Image.LANCZOS), dtype=float)
        cs.append((np.corrcoef(Aa.ravel(), Bb.ravel())[0, 1], np.corrcoef(Aa.ravel(), Bb[::-1].ravel())[0, 1]))
    check(all(c > 0.97 for c, _ in cs), f"G{g}: patches rendered like the installed ones (min r {min(c for c, _ in cs):.4f}; "
          f"upside-down mean {np.mean([f for _, f in cs]):.3f})")
    npg.get_pixmap(dpi=110).save(str(HERE / f"preview_{name}_110dpi.png"))
    return npg.rect.height


hs = [build(g) for g in (1, 3, 5)]
check(len({round(h, 2) for h in hs}) == 1, f"the three grids have the same height {[round(h, 2) for h in hs]}")
print(f"each file: {W_BP:.2f} x {hs[0]:.2f} bp = {W_BP / 72 * 2.54:.2f} x {hs[0] / 72 * 2.54:.2f} cm at "
      f"\\includegraphics[width=\\linewidth] in a {SUB_FRAC}\\textwidth subfigure")
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED:\n  " + "\n  ".join(FAIL))
sys.exit(1 if FAIL else 0)
