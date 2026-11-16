# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/fig_checks.py
# Thesis output : shared checks of the matplotlib figure scripts (printed font size, fonts, words and numbers identical to the thesis figure, text inside the page, no overlap)
# Inputs        : the new figure PDF and the thesis build's figure file under CLAM_TR_PROJECT_DIR/thesis/main/figures
# Outputs       : previews/ and checks/ next to this module
# Run           : imported by the make_fig*.py scripts of this folder
# Provenance    : adapted from the thesis figure scripts; the checks are unchanged. Lines marked "# repo path"
#                 read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Shared checks for the matplotlib data figures of the thesis.

Every figure is drawn on a canvas whose width is its printed width (frac x \\textwidth), so the print
scale is 1.0 and the native font size is the printed size. check_figure() measures the NEW PDF and compares it with
the CURRENT thesis PDF:
  * printed text sizes (span size x scale): min / median, every span < 9.95 pt listed (with a superscript test)
  * font families found
  * words and number tokens identical (multisets) old vs new
  * every text line inside the page, no text line overlapping another (0.08 em tolerance)
  * printed width / height in cm
It also writes a 110-dpi preview of the printed size and returns a dict with the measured values.
"""
import collections
import json
import os  # repo path (reads the environment variables of this repository)
import pathlib
import re
import statistics

import fitz

ROOT = pathlib.Path(os.environ.get("CLAM_TR_PROJECT_DIR", pathlib.Path(__file__).resolve().parents[2]))  # repo path
FIG = ROOT / "thesis" / "main" / "figures"
HERE = pathlib.Path(__file__).resolve().parent
PREV = HERE / "previews"
TW = 441.01773 * 72 / 72.27          # \textwidth in bp
TEXTHEIGHT_CM = 24.2
FLOOR = 9.95
NUM = re.compile(r"(?<![A-Za-z0-9])[+\-\u2212\u2013]?\d+(?:[.,]\d+)*")
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)
    return cond


def spans(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    out.append((s, l))
    return out


def check_figure(name, frac, new_pdf):
    """Run the checks of the module docstring on new_pdf (printed at frac x \\textwidth)."""
    old = fitz.open(FIG / f"{name}.pdf")[0]
    new = fitz.open(new_pdf)[0]
    printed_w = frac * TW
    s = printed_w / new.rect.width
    s_old_frac = {"fig_3_2_6_forest_plot": 0.85, "fig_2_X_epoch_distribution": 0.85,
                  "fig_3_X_failure_mode_rank1": 0.95, "fig_calibration_reliability": 0.72,
                  "fig_4_X_multiseed_power": 0.85, "fig_4_Y_v16_vs_v20_power": 0.85}[name]
    s_old = s_old_frac * TW / old.rect.width
    res = {"name": name, "frac": frac, "natural_bp": (round(new.rect.width, 2), round(new.rect.height, 2)),
           "scale": round(s, 4), "printed_cm": (round(printed_w / 72 * 2.54, 2), round(new.rect.height * s / 72 * 2.54, 2))}
    print(f"== {name}: natural {new.rect.width:.2f} x {new.rect.height:.2f} bp, printed at {frac} \\textwidth "
          f"-> scale {s:.4f}, printed {res['printed_cm'][0]} x {res['printed_cm'][1]} cm")
    # 1. printed sizes
    sp = spans(new)
    sizes = [x["size"] * s for x, _ in sp]
    small = []
    for x, l in sp:
        if x["size"] * s < FLOOR:
            # superscript / subscript test: span flags bit 0 (superscript) or a span clearly smaller than the
            # largest span of its own line and raised/lowered from the line's baseline
            big = max(y["size"] for y in l["spans"])
            script = bool(x["flags"] & 1) or (x["size"] < 0.85 * big and abs(x["origin"][1] - l["spans"][0]["origin"][1]) > 0.5)
            small.append((x["text"], round(x["size"] * s, 2), script))
    fonts = sorted({x["font"] for x, _ in sp})
    old_sizes = [x["size"] * s_old for x, _ in spans(old)]
    res.update({"min": round(min(sizes), 2), "median": round(statistics.median(sizes), 2), "small": small,
                "fonts": fonts, "old_min": round(min(old_sizes), 2), "old_median": round(statistics.median(old_sizes), 2),
                "n_spans": len(sizes)})
    check(all(sc for _, _, sc in small), f"{name}: printed text min {min(sizes):.2f} pt / median "
          f"{statistics.median(sizes):.2f} pt (old {min(old_sizes):.2f} / {statistics.median(old_sizes):.2f}); "
          f"non-script spans < {FLOOR}: {[t for t in small if not t[2]]}")
    check(all(("Times" in f) or ("STIX" in f) for f in fonts), f"{name}: fonts {fonts}")
    # 2. token identity
    ot, nt = old.get_text(), new.get_text()
    ow, nw = collections.Counter(ot.split()), collections.Counter(nt.split())
    on, nn = collections.Counter(NUM.findall(ot)), collections.Counter(NUM.findall(nt))
    res["words_ok"], res["nums_ok"] = ow == nw, on == nn
    res["n_words"], res["n_nums"] = sum(ow.values()), sum(on.values())
    res["word_diff"] = {"old_only": dict(ow - nw), "new_only": dict(nw - ow)}
    res["num_diff"] = {"old_only": dict(on - nn), "new_only": dict(nn - on)}
    check(ow == nw, f"{name}: words identical ({sum(ow.values())})" + ("" if ow == nw else f" {res['word_diff']}"))
    check(on == nn, f"{name}: number tokens identical ({sum(on.values())})" + ("" if on == nn else f" {res['num_diff']}"))
    # layout: inside page, no overlap
    lines = [(" ".join(x["text"] for x in l["spans"]).strip(), fitz.Rect(l["bbox"]), max(x["size"] for x in l["spans"]))
             for b in new.get_text("dict")["blocks"] for l in b.get("lines", [])
             if "".join(x["text"] for x in l["spans"]).strip()]
    pr = new.rect
    out = [t for t, rb, _ in lines if rb.x0 < -0.3 or rb.y0 < -0.3 or rb.x1 > pr.width + 0.3 or rb.y1 > pr.height + 0.3]
    check(not out, f"{name}: all text inside the page {out}")
    # rotated lines (dir not horizontal): test their true rotated quads (separating-axis test), each quad
    # shrunk by 0.04 em at top and bottom -- the same 0.08 em tolerance as for the horizontal lines
    rawl = [l for b in new.get_text("dict")["blocks"] for l in b.get("lines", [])
            if "".join(x["text"] for x in l["spans"]).strip()]
    assert len(rawl) == len(lines)
    rot = [abs(l["dir"][1]) > 1e-3 for l in rawl]

    def quad(l):
        q = fitz.recover_line_quad(l)
        e = 0.04 * max(x["size"] for x in l["spans"])
        pts = [q.ul, q.ur, q.lr, q.ll]
        up = (q.ul - q.ll)
        n = abs(up)
        if n > 2 * e:
            u = up / n * e
            pts = [q.ul - u, q.ur - u, q.lr + u, q.ll + u]
        return [(p.x, p.y) for p in pts]

    def sat(pa, pb):
        for poly in (pa, pb):
            for k in range(4):
                (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % 4]
                ax_, ay_ = y1 - y2, x2 - x1
                pa_ = [ax_ * x + ay_ * y for x, y in pa]
                pb_ = [ax_ * x + ay_ * y for x, y in pb]
                if max(pa_) <= min(pb_) or max(pb_) <= min(pa_):
                    return False
        return True

    ov = []
    for i, (ta, ra, sa) in enumerate(lines):
        for j in range(i + 1, len(lines)):
            tb, rb, sb = lines[j]
            if rot[i] or rot[j]:
                if sat(quad(rawl[i]), quad(rawl[j])):
                    ov.append((ta, tb, "rotated quads intersect"))
                continue
            inter = fitz.Rect(ra) & rb
            if not inter.is_empty and inter.width > 0.3 and inter.height > 0.08 * min(sa, sb):
                ov.append((ta, tb, round(inter.width, 2), round(inter.height, 2)))
    res["overlaps"] = ov
    check(not ov, f"{name}: no text line overlaps another ({len(lines)} lines) {ov}")
    # 4. fit
    check(frac <= 1.0 + 1e-9 and res["printed_cm"][1] <= 0.75 * TEXTHEIGHT_CM,
          f"{name}: printed {res['printed_cm'][0]} x {res['printed_cm'][1]} cm (height limit {0.75 * TEXTHEIGHT_CM:.1f} cm)")
    PREV.mkdir(exist_ok=True)
    z = 110 * s / 72
    pm = new.get_pixmap(matrix=fitz.Matrix(z, z))
    pm.set_dpi(110, 110)
    pm.save(str(PREV / f"{name}_print110dpi.png"))
    (HERE / "checks").mkdir(exist_ok=True)
    (HERE / "checks" / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def times_rc(plt):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "clam-tr",
    })


def summary():
    print()
    print("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
    return not FAILS
