#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig15_failure_mode_rank1.py
# Thesis output : Figure 15 (fig_3_X_failure_mode_rank1)
# Inputs        : attention_fold0_per_slide.csv and attention_fold0_results.json written by analysis/attention_fold0_statistics.py (per-slide, not in the repository); Each figure is compared at the end with the figure file of the thesis build (CLAM_TR_PROJECT_DIR/thesis/main/figures) by fig_checks.check_figure
# Outputs       : PDF + PNG next to this script
# Run           : python -P 06-thesis-figures/plots/make_fig15_failure_mode_rank1.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 15 (fig:failure_mode_rank1, figure file fig_3_X_failure_mode_rank1.pdf) with every text item
>= 10 pt printed. Writes fig_3_X_failure_mode_rank1.{pdf,png} next to this script.

The previous version of this figure was drawn directly from the per-slide attention dump. Its per-slide values are
saved by analysis/attention_fold0_statistics.py in attention_fold0_per_slide.csv (columns label, pred_UNI_RetNet =
authoritative prediction, rank1_UNI_RetNet_new = max of the stored softmax over all patches, i.e. the value the
figure plots); this script plots from that CSV.
Gates (asserted): 1,803 slides, unique ids; correct / wrong counts, both medians, the one-sided Mann-Whitney p, and
for every grade the counts and both medians = attention_fold0_results.json (|diff| < 1e-12), exactly as the
previous generator asserted; the bar heights and the printed numbers come from these values; fig_checks.
Changes relative to the previous version (layout and fonts only): canvas = printed size (1.0 \textwidth, scale
1.0); the two panels are stacked (a) above (b) instead of side by side, so that the 12 count labels of (b) fit above
their bars at 10 pt; the statistics box of (a) moved to the empty upper middle (line spacing 1.5); all text
10 pt (panel letters 10.5 pt bold), Times New Roman + stix math, as before.
Usage: python -P make_fig15_failure_mode_rank1.py
"""
import json
import os  # repo path (reads the environment variables of this repository)
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_checks as C  # noqa: E402  # repo path (module renamed)

NAME, FRAC = "fig_3_X_failure_mode_rank1", 1.0
D15 = pathlib.Path(os.environ.get("CLAM_TR_ATTENTION_STATS_DIR", HERE.parent / "analysis"))  # repo path (outputs of analysis/attention_fold0_statistics.py)
CSV = D15 / "attention_fold0_per_slide.csv"
RESULTS = D15 / "attention_fold0_results.json"
N_SLIDES = 1803
N_CLASSES = 6
GRADE_LABEL = ["Benign", "ISUP 1", "ISUP 2", "ISUP 3", "ISUP 4", "ISUP 5"]
C_CORRECT_FILL, C_CORRECT_EDGE = "#9ECAE1", "#08519C"
C_WRONG = "#B2182B"
FS = 10

plt.rcdefaults()
C.times_rc(plt)
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.titlesize": 10.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def fmt_p(p):
    return f"{p:.2f}" if p >= 0.01 else f"{p:.1e}"


def main():
    df = pd.read_csv(CSV)
    assert len(df) == N_SLIDES and df.slide_id.is_unique
    r1 = df["rank1_UNI_RetNet_new"].to_numpy(dtype=np.float64)
    pred = df["pred_UNI_RetNet"].to_numpy()
    lab = df["label"].to_numpy()
    assert np.isfinite(r1).all() and (r1 > 0).all() and (r1 <= 1).all()
    correct = pred == lab
    rc, rw = r1[correct], r1[~correct]
    p = float(mannwhitneyu(rc, rw, alternative="greater").pvalue)
    med_c, med_w = float(np.median(rc)), float(np.median(rw))
    R = json.loads(RESULTS.read_text())["B_UNI_RetNet_bestHP_fold0_failure_mode"]
    assert (R["n_correct"], R["n_wrong"]) == (rc.size, rw.size)
    assert abs(R["median_correct"] - med_c) < 1e-12 and abs(R["median_wrong"] - med_w) < 1e-12
    assert abs(R["mwu_p_one_sided_correct_gt_wrong"] - p) < 1e-12
    print(f"gate: n_correct {rc.size}, n_wrong {rw.size}, medians {med_c} / {med_w}, p {p} = attention_fold0_results.json")

    W = FRAC * C.TW / 72
    fig, axes = plt.subplots(2, 1, figsize=(W, 5.9), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, hspace=0.04)

    # ---- (a) rank-1 mass by correctness ----
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    hc, _, _ = ax.hist(rc, bins=bins, density=True, color=C_CORRECT_FILL, edgecolor=C_CORRECT_EDGE,
                       linewidth=0.4, label=f"Correct ($n$ = {rc.size:,})")
    hw, _, _ = ax.hist(rw, bins=bins, density=True, histtype="step", color=C_WRONG, linewidth=1.1,
                       label=f"Wrong ($n$ = {rw.size:,})")
    ax.axvline(med_c, color=C_CORRECT_EDGE, linestyle=(0, (4.5, 2.0)), linewidth=0.9)
    ax.axvline(med_w, color=C_WRONG, linestyle=(0, (2.0, 1.2)), linewidth=1.1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.36 * max(hc.max(), hw.max()))
    ax.set_xlabel("Rank-1 attention mass")
    ax.set_ylabel("Density")
    leg = ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.3)
    # statistics box in the empty upper middle (at the old place, right of the legend's foot, it would sit on the
    # bars at 0.6-1.0 in the wider panel); line spacing 1.5 because the stix italic p has a taller line box
    ax.text(0.695, 0.965,
            f"Median (correct) = {med_c:.3f}\nMedian (wrong) = {med_w:.3f}\n"
            f"Mann–Whitney $p$ = {fmt_p(p)}\n(one-sided, correct > wrong)",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS, linespacing=1.5,
            bbox=dict(facecolor="white", edgecolor="0.6", linewidth=0.5, boxstyle="round,pad=0.3"))
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_title("(a)", loc="left", fontweight="bold")

    # ---- (b) median rank-1 mass per true grade x correctness ----
    ax = axes[1]
    x = np.arange(N_CLASSES)
    w = 0.38
    med_cg, med_wg, n_cg, n_wg = [], [], [], []
    for k in range(N_CLASSES):
        c = r1[(lab == k) & correct]
        wr = r1[(lab == k) & ~correct]
        med_cg.append(float(np.median(c)))
        med_wg.append(float(np.median(wr)))
        n_cg.append(c.size)
        n_wg.append(wr.size)
        row = R["per_grade"][k]
        assert row["grade"] == k
        assert (row["n_correct"], row["n_wrong"]) == (c.size, wr.size)
        assert abs(row["median_correct"] - med_cg[-1]) < 1e-12 and abs(row["median_wrong"] - med_wg[-1]) < 1e-12
    print("gate: per-grade counts and medians (6 grades) = attention_fold0_results.json")
    bc = ax.bar(x - w / 2, med_cg, w, color=C_CORRECT_FILL, edgecolor=C_CORRECT_EDGE, linewidth=0.5,
                label="Correct (median)")
    bw = ax.bar(x + w / 2, med_wg, w, color=C_WRONG, edgecolor="black", linewidth=0.5, label="Wrong (median)")
    assert np.allclose([b.get_height() for b in bc], med_cg, atol=0) and np.allclose([b.get_height() for b in bw], med_wg, atol=0)
    for i in range(N_CLASSES):
        ax.text(i - w / 2, med_cg[i] + 0.012, f"{n_cg[i]}", ha="center", va="bottom", fontsize=FS)
        ax.text(i + w / 2, med_wg[i] + 0.012, f"{n_wg[i]}", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(x)
    ax.set_xticklabels(GRADE_LABEL)
    ax.set_xlim(-0.6, N_CLASSES - 0.4)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Median rank-1 attention mass")
    ax.set_xlabel("True grade")
    ax.legend(loc="upper right", frameon=False, handlelength=1.4, borderaxespad=0.3)
    ax.grid(alpha=0.25, linewidth=0.4, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("(b)", loc="left", fontweight="bold")

    pdf, png = HERE / f"{NAME}.pdf", HERE / f"{NAME}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    C.check_figure(NAME, FRAC, pdf)
    C.summary()


if __name__ == "__main__":
    main()
