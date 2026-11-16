#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_figs_12_17_18.py
# Thesis output : Figure 12 (fig_3_2_6_forest_plot), Figure 17 (fig_4_X_multiseed_power) and Figure 18 (fig_4_Y_v16_vs_v20_power)
# Inputs        : the data computed by analysis/fold_aligned_contrasts_figs12_17_18.py (imported); Each figure is compared at the end with the figure file of the thesis build (CLAM_TR_PROJECT_DIR/thesis/main/figures) by fig_checks.check_figure
# Outputs       : PDF + PNG next to this script
# Run           : python -P 06-thesis-figures/plots/make_figs_12_17_18.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figures 12, 17 and 18 with every text item >= 10 pt printed, Times New Roman. Writes
{fig_3_2_6_forest_plot, fig_4_X_multiseed_power, fig_4_Y_v16_vs_v20_power}.{pdf,png} next to this script.

Data: analysis/fold_aligned_contrasts_figs12_17_18.py
is IMPORTED for its data (fold-aligned bestHP QWKs, contrasts, bootstrap and projection intervals; importing writes
nothing). Drawing code as in the previous version of the three figures, with these changes only:
  * the canvas is the PRINTED size (frac x \textwidth wide, scale 1.0), so native font size = printed size;
  * fonts: Times New Roman, mathtext stix; ticks / legend / in-axes text 10 pt, axis titles 10.5 pt, titles 11 pt;
  * Fig. 18: legend placed in the empty lower-right corner (was loc="best"); multi-line labels with math
    (Fig. 17 y title, Fig. 18 x tick labels) at line spacing 1.4 so that their line boxes do not overlap.
Asserted: plotted marker positions and whisker ends = computed values; fig_checks (words and number tokens identical
to the current thesis figures, printed sizes >= 10 pt, text inside the page, no text overlap, fits the page).
Usage: python -P make_figs_12_17_18.py
"""
import importlib.util
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_checks as C  # noqa: E402  # repo path (module renamed)

GEN = HERE.parent / "analysis" / "fold_aligned_contrasts_figs12_17_18.py"  # repo path
spec = importlib.util.spec_from_file_location("G14b", GEN)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)                      # data only
TICK, LABEL, TITLE = 10, 10.5, 11


def U(s):
    """The three math symbols of these labels as Unicode characters of Times New Roman (same code points as the
    mathtext glyphs of the current figures: U+0394, U+2212, U+00D7); keeps one font and normal line boxes."""
    out = s.replace(r"$\Delta$", "Δ").replace("$-$", "−").replace(r"$\times$", "×")
    assert "$" not in out and "\\" not in out, out
    return out


def setup(frac, h_in):
    G.apply_style()                             # seaborn whitegrid + notebook rc (as the current figures)
    C.times_rc(plt)
    plt.rcParams["savefig.bbox"] = None         # canvas = printed size
    return plt.figure(figsize=(frac * C.TW / 72, h_in))


def finish(fig, name, frac):
    fig.tight_layout(pad=0.3)
    pdf, png = HERE / f"{name}.pdf", HERE / f"{name}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    C.check_figure(name, frac, pdf)


# --------------------------------------------------------------------------------------- Fig. 12
def fig12():
    name, frac = "fig_3_2_6_forest_plot", 1.0
    fig = setup(frac, 5.6)
    ax = fig.add_subplot(111)
    rows = G.F12
    y = np.arange(len(rows))[::-1]
    labels, rendered = [], []
    for i, r in enumerate(rows):
        color = G.COL_UNI if r["encoder"] == "UNI" else G.COL_UNI2H
        spans = r["boot95_spans_zero"]
        marker = "s" if spans else "o"
        lo, hi = r["boot95_v20"]
        cont = ax.errorbar(r["mean"], y[i], xerr=[[r["mean"] - lo], [hi - r["mean"]]], fmt=marker, color=color,
                           capsize=4, markersize=8, markerfacecolor="white" if spans else color, markeredgewidth=1.5)
        rendered.append((r, G.errorbar_rendered(cont)))
        lab = dict((n, l) for n, l, _ in G.F12_DEFS)[r["contrast"]]
        shown = "Difference of differences" if r["contrast"] == "Interaction" else r["contrast"]
        labels.append(f"{r['encoder']}: {shown}\n{G.true_minus(lab)}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.set_xlabel(U(r"$\Delta$ QWK (paired difference) with bootstrap 95% CI"), fontsize=LABEL)
    ax.set_title("Factorial decomposition: per-encoder contrasts", fontsize=TITLE)
    ax.grid(axis="x", alpha=0.3)
    filled = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=7, label="CI excludes zero")
    hollow = mlines.Line2D([], [], color="black", marker="s", linestyle="None", markersize=7,
                           markerfacecolor="white", markeredgewidth=1.5, label="CI spans zero")
    ax.legend(handles=[filled, hollow], loc="upper left", framealpha=0.9, fontsize=TICK)
    for r, rr in rendered:
        assert abs(rr["x"] - r["mean"]) < 1e-12 and np.allclose([rr["seg"][0][0], rr["seg"][1][0]], r["boot95_v20"], atol=1e-12)
    print(f"{name}: 8 markers and whiskers = computed means and bootstrap intervals")
    finish(fig, name, frac)


# --------------------------------------------------------------------------------------- Fig. 17
def fig17():
    name, frac = "fig_4_X_multiseed_power", 1.0
    fig = setup(frac, 3.6)
    ax = fig.add_subplot(111)
    mean = G.F18["mean"]
    top = -np.inf
    for pr in G.F18["projection"]:
        n = pr["n"]
        lo, hi = pr["mc95"]
        color = G.COL_UNI2H if n == 5 else G.COL_UNI
        cont = ax.errorbar([n], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=4, markersize=8)
        rr = G.errorbar_rendered(cont)
        assert abs(rr["y"] - mean) < 1e-12 and np.allclose([rr["seg"][0][1], rr["seg"][1][1]], pr["mc95"], atol=1e-12)
        ax.text(n, hi + 0.0008, f"n={n}", ha="center", fontsize=TICK)
        top = max(top, hi + 0.0008)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(U(r"Effective sample size (5 folds $\times$ N seeds)"), fontsize=LABEL)
    ax.set_ylabel(U("$\\Delta$ QWK\n(SpatialMultRetNet $-$ ABMIL on UNI2-h)"), fontsize=LABEL, linespacing=1.4)
    ax.set_xscale("log")
    ax.set_xticks(G.NS)
    ax.set_xticklabels(G.NS)
    ax.tick_params(axis="both", labelsize=TICK)
    ax.minorticks_off()
    lo_lim, hi_lim = ax.get_ylim()
    ax.set_ylim(lo_lim, max(hi_lim, top + 0.0022))
    ax.set_yticks(np.arange(0, 0.0176, 0.0025))          # the ticks the current figure prints (0.0000-0.0175)
    ax.set_title("Normal-approximation CI projection vs effective sample size", fontsize=TITLE)
    print(f"{name}: 5 markers and whiskers = computed projection intervals")
    finish(fig, name, frac)


# --------------------------------------------------------------------------------------- Fig. 18
def fig18():
    name, frac = "fig_4_Y_v16_vs_v20_power", 1.0
    fig = setup(frac, 3.9)
    ax = fig.add_subplot(111)
    cat16 = U("600-slide development cohort\nconcatenation model without retention\n$-$ non-gated ABMIL")
    cat20 = U("9,128-slide cohort\nSpatialMultRetNet $-$ ABMIL")
    for key, cat, color, lab in (("v16", cat16, G.COL_UNI2H, "600-slide development cohort"),
                                 ("v20", cat20, G.COL_UNI, "9,128-slide cohort")):
        m = G.F19[key]["mean"]
        lo, hi = G.F19[key]["boot95_v20"]
        cont = ax.errorbar([cat], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=color, capsize=5, markersize=10, label=lab)
        rr = G.errorbar_rendered(cont)
        assert abs(rr["y"] - m) < 1e-12 and np.allclose([rr["seg"][0][1], rr["seg"][1][1]], [lo, hi], atol=1e-12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(U(r"$\Delta$ QWK (model $-$ ABMIL baseline on UNI)"), fontsize=LABEL)
    ax.set_title(U(r"Model $-$ ABMIL contrasts in the 600-slide and 9,128-slide cohorts"), fontsize=TITLE)
    ax.tick_params(axis="both", labelsize=TICK)
    ax.set_yticks(np.arange(-0.005, 0.01001, 0.0025))     # the ticks the current figure prints (-0.0050-0.0100)
    ax.legend(loc="lower right", fontsize=TICK)               # empty corner (upper left covered the v16 whisker)
    for tl in ax.get_xticklabels():
        tl.set_linespacing(1.4)                              # stix math lines have taller line boxes
    print(f"{name}: 2 markers and whiskers = computed means and bootstrap intervals")
    finish(fig, name, frac)


fig12()
fig17()
fig18()
C.summary()
