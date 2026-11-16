#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_appendix_figs_1_2_3.py
# Thesis output : Appendix Figures 1, 2 and 3 (fig_lora_factorial_bars, fig_lora_vs_frozen_forest, fig_lora_stage2_folds)
# Inputs        : 05-experimental/lora_v1_results/*.json; data/lora_planned_tests_results.json; the thesis build's figure files under CLAM_TR_PROJECT_DIR/thesis/main/figures for the final comparison
# Outputs       : PDF (+ PNG) and previews next to this script
# Run           : python -P 06-thesis-figures/plots/make_appendix_figs_1_2_3.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Appendix Figures 1-3 (A.1-A.3 below) with every text item >= 10 pt printed (KTU thesis guide,
section 5.1). Writes the PDF and PNG files and preview images next to this script.

  A.1 fig_lora_factorial_bars.pdf    Stage 4 means of the twelve LoRA-UNI configurations (drawn as a dot plot)
  A.2 fig_lora_vs_frozen_forest.pdf  frozen UNI vs LoRA-UNI, paired forest plot of the six aggregators
  A.3 fig_lora_stage2_folds.pdf      Stage 2 fold QWKs (dot plot)

Same data files, same data assertions, same texts, colours, markers, axis ranges and ticks as the generators of the
previous versions of these figures.
Changes: Times New Roman (+ stix math) instead of DejaVu Serif; every figure drawn at its printed size (natural width =
printed width, scale 1); text sizes: ticks / annotations / legends 10.0 pt, axis labels and titles 10.5 pt (previous
versions: 7.2-8.5);
A.1 and A.2 printed at \textwidth (was 0.95 \textwidth), A.3 stays at 0.85 \textwidth; heights grown to fit; marker /
line sizes of A.2 converted to printed size (they were drawn on a 711 bp canvas scaled by 0.5865); A.1 legend in one
column (the two labels no longer fit side by side at 10 pt).
Asserted per figure: page width = printed width; words and number tokens identical to the installed thesis figure;
every text span >= 9.95 pt; no text line overlaps another; no text and no drawing outside the page; legend does not
cover any data marker or text; printed height <= 18 cm; marker / whisker / line positions = the data.
Usage: python -P make_appendix_figs_1_2_3.py
"""
import collections
import json
import os  # repo path (reads the environment variables of this repository)
import pathlib
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fitz  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("CLAM_TR_PROJECT_DIR", pathlib.Path(__file__).resolve().parents[2]))  # repo path
FIG = ROOT / "thesis/main/figures"
NBDIR = HERE.parents[1] / "05-experimental" / "lora_v1_results"  # repo path
TESTS = HERE.parent / "data" / "lora_planned_tests_results.json"  # repo path
TW = 441.01773 * 72 / 72.27                       # \textwidth in bp (main.log)
FS, FS_LAB, FS_TITLE = 10.0, 10.5, 10.5
MIN_PRINTED = 9.95
FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def style():
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Times New Roman"], "mathtext.fontset": "stix", "font.size": FS,
                         "axes.labelsize": FS_LAB, "axes.titlesize": FS_TITLE, "xtick.labelsize": FS,
                         "ytick.labelsize": FS, "legend.fontsize": FS, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5})


def legend_clear(fig, ax, leg, marker_xy):
    """Legend box vs data markers (display coords) and vs every axes text."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    hits = []
    for x, y in marker_xy:
        px, py = ax.transData.transform((x, y))
        if lb.x0 - 3 < px < lb.x1 + 3 and lb.y0 - 3 < py < lb.y1 + 3:
            hits.append(("marker", round(x, 4), round(y, 3)))
    for t in ax.texts:
        if t.get_window_extent(r).overlaps(lb):
            hits.append(("text", t.get_text()))
    return hits


def pdf_checks(name, frac, extra_txt_boxes_ok=True):
    new = fitz.open(HERE / f"{name}.pdf")[0]
    old = fitz.open(FIG / f"{name}.pdf")[0]
    pw = frac * TW
    check(abs(new.rect.width - pw) < 0.05, f"{name}: page width {new.rect.width:.2f} bp = {frac} \\textwidth "
          f"({pw:.2f}); scale 1.0000; printed {pw / 72 * 2.54:.2f} x {new.rect.height / 72 * 2.54:.2f} cm "
          f"(was {frac_old[name] * TW / 72 * 2.54:.2f} x {old.rect.height * frac_old[name] * TW / old.rect.width / 72 * 2.54:.2f} cm)")
    check(new.rect.height / 72 * 2.54 <= 18.0, f"{name}: printed height <= 18 cm")
    sp = [s for b in new.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
    sz = sorted(s["size"] for s in sp)
    small = [(s["text"], round(s["size"], 2)) for s in sp if s["size"] < MIN_PRINTED]
    check(not small, f"{name}: printed text min {sz[0]:.2f} pt, median {statistics.median(sz):.2f} pt; "
          f"fonts {sorted({s['font'] for s in sp})}" + (f"; < 9.95: {small}" if small else ""))
    ot, nt = old.get_text(), new.get_text()
    ow, nw = collections.Counter(ot.split()), collections.Counter(nt.split())
    check(ow == nw, f"{name}: words identical to the installed figure ({sum(ow.values())})"
          + ("" if ow == nw else f" old-only {dict(ow - nw)} new-only {dict(nw - ow)}"))
    NUM = re.compile(r"(?<![A-Za-z0-9])[+\-\u2212\u2013]?\d+(?:[.,]\d+)*")
    on, nn = collections.Counter(NUM.findall(ot)), collections.Counter(NUM.findall(nt))
    check(on == nn, f"{name}: number tokens identical ({sum(on.values())})"
          + ("" if on == nn else f" old-only {dict(on - nn)} new-only {dict(nn - on)}"))
    W, H = new.rect.width, new.rect.height
    off = [("drawing", tuple(round(v, 1) for v in d["rect"])) for d in new.get_drawings()
           if d["rect"].x0 < -0.5 or d["rect"].y0 < -0.5 or d["rect"].x1 > W + 0.5 or d["rect"].y1 > H + 0.5]
    off += [("text", b[4][:25]) for b in new.get_text("blocks") if b[0] < -0.3 or b[1] < -0.3 or b[2] > W + 0.3 or b[3] > H + 0.3]
    check(not off, f"{name}: every drawing and text inside the page {off[:4]}")
    lines = [(" ".join(x["text"] for x in l["spans"]).strip(), fitz.Rect(l["bbox"]), max(x["size"] for x in l["spans"]))
             for b in new.get_text("dict")["blocks"] for l in b.get("lines", []) if "".join(x["text"] for x in l["spans"]).strip()]
    ov = [(a[0], b[0]) for i, a in enumerate(lines) for b in lines[i + 1:]
          if not (fitz.Rect(a[1]) & b[1]).is_empty and (fitz.Rect(a[1]) & b[1]).width > 0.3
          and (fitz.Rect(a[1]) & b[1]).height > 0.08 * min(a[2], b[2])]
    check(not ov, f"{name}: no text line overlaps another in the PDF ({len(lines)} lines) {ov}")
    for dpi, tag in ((110, "110dpi"), (200, "print200dpi")):
        pm = new.get_pixmap(dpi=dpi)
        pm.save(str(HERE / f"preview_{name}_{tag}.png"))


frac_old = {"fig_lora_factorial_bars": 0.95, "fig_lora_vs_frozen_forest": 0.95, "fig_lora_stage2_folds": 0.85}


# ------------------------------------------------------------------------------------------------ Fig. A.1
def figA1(frac=1.0):
    name = "fig_lora_factorial_bars"
    v = json.loads((NBDIR / "v20_lora_features_results.json").read_text(encoding="utf-8"))
    sorted_configs = sorted(v.items(), key=lambda kv: kv[1]["mean_qwk"], reverse=True)
    names = [k for k, _ in sorted_configs]
    means = [vv["mean_qwk"] for _, vv in sorted_configs]
    stds = [statistics.stdev(vv["fold_qwks"]) for _, vv in sorted_configs]
    for (k, vv), m in zip(sorted_configs, means):
        assert len(vv["fold_qwks"]) == 5 and abs(statistics.mean(vv["fold_qwks"]) - m) < 1e-9, k
    u2 = json.loads(TESTS.read_text(encoding="utf-8"))["per_fold"]["frozen_uni2h_SpatialMultRetNet_bestHP"]
    U2_SD = statistics.stdev(u2)
    assert f"{statistics.mean(u2):.4f}" == "0.9530" and f"{U2_SD:.4f}" == "0.0033", (statistics.mean(u2), U2_SD)
    assert names[0] == "SpatialAdd_defaultHP" and f"{means[0]:.4f}" == "0.9529" and f"{stds[0]:.4f}" == "0.0026"
    baseline_uni2h = 0.9530
    style()
    W, H = frac * TW, 340.0
    fig, ax = plt.subplots(figsize=(W / 72, H / 72))
    ypos = list(range(len(names)))[::-1]
    ax.axvspan(baseline_uni2h - U2_SD, baseline_uni2h + U2_SD, color="red", alpha=0.10, lw=0,
               label="Baseline +/-1 sigma noise envelope")
    ax.axvline(baseline_uni2h, color="red", linestyle="--", linewidth=0.8,
               label=f"Frozen UNI2-h SpatialMultRetNet baseline ({baseline_uni2h:.4f})")
    cont = ax.errorbar(means, ypos, xerr=stds, fmt="o", color="#4C72B0", markersize=4.0, markeredgecolor="black",
                       markeredgewidth=0.4, ecolor="#4C72B0", elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3)
    for x, sg, m, sd in zip(cont.lines[0].get_xdata(), cont.lines[2][0].get_segments(), means, stds):
        assert abs(x - m) < 1e-12 and abs(sg[0][0] - (m - sd)) < 1e-12 and abs(sg[1][0] - (m + sd)) < 1e-12
    for y, m, s in zip(ypos, means, stds):
        ax.text(m + s + 0.0008, y, f"{m:.4f}", va="center", fontsize=FS, zorder=4,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(names) - 0.3)
    xl = ax.set_xlabel("5-fold mean QWK (+/-1 sigma)")
    ax.set_xlim(0.925, 0.962)
    ax.set_xticks([0.925, 0.930, 0.935, 0.940, 0.945, 0.950, 0.955, 0.960])
    ax.set_title("Stage 4: LoRA-UNI features through the six aggregators\n(12 configurations, sorted by mean QWK)")
    h, lab = ax.get_legend_handles_labels()
    leg = fig.legend(h[::-1], lab[::-1], loc="lower center", ncol=1, frameon=False, borderpad=0.2, handlelength=1.8,
                     bbox_to_anchor=(0.5, 0.0))
    ax.grid(axis="x", linestyle=":", alpha=0.5, linewidth=0.6)
    ax.grid(axis="y", linestyle=":", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.canvas.draw()
    leg_h = leg.get_window_extent(fig.canvas.get_renderer()).height * 72 / fig.dpi
    fig.tight_layout(pad=0.3, rect=(0, (leg_h + 7.5) / H, 1, 1))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    gap = (xl.get_window_extent(r).y0 - leg.get_window_extent(r).y1) * 72 / fig.dpi
    check(gap >= 3.0, f"{name}: x-axis label vs legend gap {gap:.2f} pt (>= 3)")
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts + ax.get_yticklabels() + ax.get_xticklabels()]
    over = [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[1].overlaps(b[1])]
    check(not over, f"{name}: no text overlaps another (rendered extents) {over}")
    # value labels must not cover another row's marker / whisker
    hit = []
    for t in ax.texts:
        tb = t.get_window_extent(r)
        for y, m, s in zip(ypos, means, stds):
            x0, yy = ax.transData.transform((m - s, y))
            x1, _ = ax.transData.transform((m + s, y))
            if tb.y0 < yy < tb.y1 and not (tb.x1 < x0 or tb.x0 > x1):
                hit.append((t.get_text(), round(m, 4)))
    check(not hit, f"{name}: value labels clear of every error bar {hit}")
    fig.savefig(HERE / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(HERE / f"{name}.png", dpi=600)
    plt.close(fig)
    pdf_checks(name, frac)


# ------------------------------------------------------------------------------------------------ Fig. A.2
def figA2(frac=1.0):
    name = "fig_lora_vs_frozen_forest"
    PF = json.loads(TESTS.read_text(encoding="utf-8"))["per_fold"]
    v = json.loads((NBDIR / "v20_lora_features_results.json").read_text(encoding="utf-8"))
    aggs = ["ABMIL", "DualBranch", "RetNet", "SpatialAdd", "SpatialMult", "SpatialMultRetNet"]
    hard = {"ABMIL": 0.9456, "DualBranch": 0.9499, "RetNet": 0.9543, "SpatialAdd": 0.9494, "SpatialMult": 0.9386,
            "SpatialMultRetNet": 0.9519}
    frozen = {}
    for a in aggs:
        f = PF["frozen_uni"][f"{a}_bestHP"]
        assert len(f) == 5 and f"{statistics.mean(f):.4f}" == f"{hard[a]:.4f}", a
        frozen[a] = (statistics.mean(f), statistics.stdev(f))
    lora_means = [v[f"{a}_bestHP"]["mean_qwk"] for a in aggs]
    lora_stds = [statistics.stdev(v[f"{a}_bestHP"]["fold_qwks"]) for a in aggs]
    frozen_means = [frozen[a][0] for a in aggs]
    frozen_stds = [frozen[a][1] for a in aggs]
    style()
    W, H = frac * TW, 255.0
    fig, ax = plt.subplots(figsize=(W / 72, H / 72))
    ypos = list(range(len(aggs)))
    offset = 0.18
    # the previous version drew markersize 8 / capsize 4 / linewidth 1.5 on a 711.65 bp canvas printed at 0.5865 -> printed values
    c_f = ax.errorbar(frozen_means, [y - offset for y in ypos], xerr=frozen_stds, fmt="o", color="#4C72B0",
                      label="Frozen UNI", capsize=2.5, markersize=5.0, linewidth=1.0)
    c_l = ax.errorbar(lora_means, [y + offset for y in ypos], xerr=lora_stds, fmt="s", color="#C44E52",
                      label="LoRA UNI (Stage 4)", capsize=2.5, markersize=5.0, linewidth=1.0)
    for cont, means, sds in ((c_f, frozen_means, frozen_stds), (c_l, lora_means, lora_stds)):
        xs = cont.lines[0].get_xdata()
        segs = cont.lines[2][0].get_segments()
        assert len(segs) == 6
        for x, sg, m, sd in zip(xs, segs, means, sds):
            assert abs(x - m) < 1e-12 and abs(sg[0][0] - (m - sd)) < 1e-12 and abs(sg[1][0] - (m + sd)) < 1e-12
    check(True, f"{name}: 12 markers at the means, whiskers = +/- sample SD")
    for y, f_m, l_m in zip(ypos, frozen_means, lora_means):
        delta = l_m - f_m
        sign = "+" if delta >= 0 else ""
        ax.text(0.965, y, f"{sign}{delta:.4f}", fontsize=FS, va="center", color="#888888")
    ax.set_yticks(ypos)
    ax.set_yticklabels(aggs)
    ax.set_xlabel("5-fold mean QWK (+/-1 sigma)")
    ax.set_xlim(0.915, 0.975)
    ax.set_xticks([0.92, 0.93, 0.94, 0.95, 0.96, 0.97])
    ax.set_title("Frozen UNI vs LoRA-UNI feature pipeline -- paired comparison (bestHP)")
    ax.axvline(0.95, color="gray", linestyle=":", alpha=0.4)
    leg = ax.legend(loc="lower left", fontsize=FS)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.5, len(aggs) - 0.1)
    ax.text(0.965, len(aggs) - 0.38, "delta QWK", fontsize=FS, va="center", ha="left", color="#555555",
            fontweight="bold")
    fig.tight_layout(pad=0.3)
    pts = [(m, y - offset) for m, y in zip(frozen_means, ypos)] + [(m, y + offset) for m, y in zip(lora_means, ypos)]
    pts += [(m - s, y - offset) for m, s, y in zip(frozen_means, frozen_stds, ypos)]
    pts += [(m + s, y - offset) for m, s, y in zip(frozen_means, frozen_stds, ypos)]
    pts += [(m - s, y + offset) for m, s, y in zip(lora_means, lora_stds, ypos)]
    pts += [(m + s, y + offset) for m, s, y in zip(lora_means, lora_stds, ypos)]
    hits = legend_clear(fig, ax, leg, pts)
    check(not hits, f"{name}: legend covers no marker, whisker end or text {hits}")
    r = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts + ax.get_yticklabels() + ax.get_xticklabels()]
    over = [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[1].overlaps(b[1])]
    # delta values must stay right of every whisker
    dx = min(t.get_window_extent(r).x0 for t in ax.texts)
    wmax = max(ax.transData.transform((m + s, 0))[0] for m, s in
               list(zip(frozen_means, frozen_stds)) + list(zip(lora_means, lora_stds)))
    check(not over and dx > wmax + 3, f"{name}: no text overlap; delta column {(dx - wmax) * 72 / fig.dpi:.1f} pt right of "
          f"the rightmost whisker {over}")
    fig.savefig(HERE / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(HERE / f"{name}.png", dpi=600)
    plt.close(fig)
    pdf_checks(name, frac)


# ------------------------------------------------------------------------------------------------ Fig. A.3
def figA3(frac=0.85):
    name = "fig_lora_stage2_folds"
    rj = json.loads((NBDIR / "lora_training_results.json").read_text(encoding="utf-8"))
    fold_qwks = [fm["qwk"] for fm in rj["fold_metrics"]]
    fold_ids = [f"Fold {fm['fold']}" for fm in rj["fold_metrics"]]
    mean_qwk, std_qwk, best_fold = rj["mean_qwk"], rj["std_qwk"], rj["best_fold"]
    assert f"{mean_qwk:.4f}" == "0.9563" and f"{std_qwk:.4f}" == "0.0027" and best_fold == 3
    assert abs(statistics.pstdev(fold_qwks) - std_qwk) < 1e-9 and abs(statistics.mean(fold_qwks) - mean_qwk) < 1e-12
    assert max(range(5), key=lambda i: fold_qwks[i]) == best_fold
    style()
    W, H = frac * TW, 290.0
    fig, ax = plt.subplots(figsize=(W / 72, H / 72))
    xpos = list(range(len(fold_ids)))
    ax.axhspan(mean_qwk - std_qwk, mean_qwk + std_qwk, color="red", alpha=0.12, lw=0, zorder=1,
               label=f"+/-1 sigma ({std_qwk:.4f})")
    ax.axhline(mean_qwk, color="red", linestyle="--", linewidth=0.8, zorder=2, label=f"5-fold mean ({mean_qwk:.4f})")
    colors = ["#FFB400" if i == best_fold else "#55A868" for i in xpos]
    pts = ax.scatter(xpos, fold_qwks, s=5.0 ** 2, c=colors, edgecolors="black", linewidths=0.4, zorder=3)
    for x, q in zip(xpos, fold_qwks):
        ax.annotate(f"{q:.4f}", (x, q), xytext=(0, 4.5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS, zorder=4)
    ax.annotate("(best)", (best_fold, fold_qwks[best_fold]), xytext=(0, 4.5 + FS * 1.25), textcoords="offset points",
                ha="center", va="bottom", fontsize=FS, color="#B86B00", fontweight="bold", zorder=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(fold_ids)
    ax.set_xlim(-0.6, len(xpos) - 0.4)
    ax.set_ylabel("Validation 5-fold QWK")
    ax.set_ylim(0.945, 0.965)
    ax.set_yticks([0.945 + 0.0025 * i for i in range(9)])
    ax.set_title("Stage 2: LoRA encoder direct classifier (5-fold cross-validation)")
    h, lab = ax.get_legend_handles_labels()
    leg = ax.legend(h[::-1], lab[::-1], loc="lower right", framealpha=0.95, borderpad=0.4, handlelength=1.8)
    ax.grid(axis="y", linestyle=":", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    offs = pts.get_offsets()
    check(all(abs(o[0] - x) < 1e-12 and abs(o[1] - q) < 1e-12 for o, x, q in zip(offs, xpos, fold_qwks)),
          f"{name}: 5 points at the fold QWKs {[round(q, 4) for q in fold_qwks]}")
    hl = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"][0]
    span = [p for p in ax.patches if p.get_alpha() is not None and abs(p.get_alpha() - 0.12) < 1e-9][0]
    ys = [span.get_y(), span.get_y() + span.get_height()]
    check(abs(hl.get_ydata()[0] - mean_qwk) < 1e-12 and abs(min(ys) - (mean_qwk - std_qwk)) < 1e-12
          and abs(max(ys) - (mean_qwk + std_qwk)) < 1e-12, f"{name}: mean line and +/- std_qwk band = JSON values")
    hits = legend_clear(fig, ax, leg, list(zip(xpos, fold_qwks)))
    check(not hits, f"{name}: legend covers no point and no text {hits}")
    r = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts + ax.get_yticklabels() + ax.get_xticklabels()
             if t.get_text().strip()]
    over = [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[1].overlaps(b[1])]
    ab = ax.get_window_extent(r)
    inside = all(t.get_window_extent(r).y1 < ab.y1 - 1 for t in ax.texts)
    check(not over and inside, f"{name}: no text overlap; value labels and '(best)' inside the axes {over}")
    fig.savefig(HERE / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)                       # no PNG twin: the thesis folder has none for A.3
    pdf_checks(name, frac)


figA1()
figA2()
figA3()
print()
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED:\n  " + "\n  ".join(FAIL))
