#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig16_calibration_reliability.py
# Thesis output : Figure 16 (fig_calibration_reliability)
# Inputs        : data/calibration_results.json (written by analysis/CLAM_TR_v20_calibration.ipynb, cell 59); Each figure is compared at the end with the figure file of the thesis build (CLAM_TR_PROJECT_DIR/thesis/main/figures) by fig_checks.check_figure
# Outputs       : PDF + PNG next to this script
# Run           : python -P 06-thesis-figures/plots/make_fig16_calibration_reliability.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 16 (fig:calibration, figure file fig_calibration_reliability.pdf), with every text item >= 10 pt
printed. Writes fig_calibration_reliability.{pdf,png} next to this script.

Data: cell 59 of analysis/CLAM_TR_v20_calibration.ipynb (Colab) re-ran the UNI2-h SpatialMultRetNet_bestHP
best-epoch checkpoints, computed the 10-bin statistics and, in the same step, wrote the first version of the figure
AND the results JSON (per_bin conf/acc/count, ECE, Brier) to the results folder
CLAM_TR_Results/v20_fullscale/UNI2-h/ (both files have the same timestamp, 4 Jun 2026 03:11; that PDF has the size
of the thesis PDF, 17,790 bytes). The JSON is data/calibration_results.json in this repository, and this script
redraws the figure from it with the cell's drawing code. No model is run.
Changes relative to the cell: Times New Roman + stix math; canvas = printed size (0.80 \textwidth, scale 1.0) so
that all text prints at >= 10 pt (the cell's figure: DejaVu 7.9-8.8 pt printed at 0.72 \textwidth).
Gates: (1) the bar heights of the CURRENT thesis PDF, read from its vector drawing, are proportional to the JSON
per-bin accuracies (so the JSON is the data behind the current figure); (2) the bars, the curve and the printed
ECE / Brier of the new figure = the JSON values; (3) the checks of fig_checks.py.
Usage: python -P make_fig16_calibration_reliability.py
"""
import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fitz  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
import fig_checks as C  # noqa: E402  # repo path (module renamed)

NAME = "fig_calibration_reliability"
FRAC = 0.80
J = json.loads((HERE.parent / "data" / "calibration_results.json").read_text(encoding="utf-8"))  # repo path
assert J["encoder"] == "UNI2-h" and J["variant"] == "SpatialMultRetNet_bestHP" and J["n_slides"] == 9128
pb = J["per_bin"]
assert len(pb) == 10 and sum(x["count"] for x in pb) == 9128
ENC, VARIANT, ece, brier = J["encoder"], J["variant"], J["ece_10bin"], J["brier_multiclass"]
# ECE recomputed from the per-bin values = stored ECE
ece_re = sum(x["count"] / 9128 * abs(x["acc"] - x["conf"]) for x in pb if x["count"])
assert abs(ece_re - ece) < 1e-6, (ece_re, ece)

# ---- gate 1: the current thesis figure's bars are these accuracies
old = fitz.open(C.FIG / f"{NAME}.pdf")[0]
bars = []
for d in old.get_drawings():
    r = d["rect"]
    if d.get("fill") is not None and d["items"] and all(it[0] in ("re", "l") for it in d["items"]) \
            and 5 < r.width < 40 and r.height >= 0:
        bars.append((round(r.x0, 2), r.height, d.get("fill"), round(r.width, 1)))
blue = [b for b in bars if b[2] and abs(b[2][0] - 0.1216) < 0.02 and abs(b[2][2] - 0.7059) < 0.02]
wmode = max(set(b[3] for b in blue), key=[b[3] for b in blue].count)
blue = sorted({(x, h) for x, h, _, w in blue if w == wmode})        # the legend handle has another width
accs = [x["acc"] or 0.0 for x in pb]
assert len(blue) == 10, blue
hs = np.array([h for _, h in blue])
ratio = hs / hs.max()
want = np.array(accs) / max(accs)
assert np.allclose(ratio, want, atol=2e-3), (ratio, want)
print("gate 1: 10 bar heights of the current PDF proportional to the JSON per-bin accuracies (max dev "
      f"{np.abs(ratio - want).max():.2e})")

# ---- figure: the calibration cell's drawing code, Times fonts, printed-size canvas
plt.rcdefaults()
C.times_rc(plt)
W = FRAC * C.TW / 72
fig, ax = plt.subplots(figsize=(W, W * 0.94))
ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
bc = ax.bar([(x['lo'] + x['hi']) / 2 for x in pb], [x['acc'] or 0 for x in pb], width=0.092, alpha=0.65,
            edgecolor="k", label="bin acc")
ln, = ax.plot([x['conf'] for x in pb if x['conf'] is not None], [x['acc'] for x in pb if x['acc'] is not None],
              "o-", color="C3", label="conf vs acc")
ax.set_xlabel("Confidence", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.tick_params(labelsize=10)
ax.set_title(f"Reliability - {ENC} {VARIANT}, pooled 5-fold OOF", fontsize=10)
ax.text(0.04, 0.92, f"ECE={ece:.3f}\nBrier={brier:.3f}", transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(0.02, 0.78))   # empty area; was lower right over the bars
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout(pad=0.3)
# gate 2: drawn values = JSON
assert np.allclose([p.get_height() for p in bc.patches], accs, atol=1e-12)
assert np.allclose(ln.get_xdata(), [x["conf"] for x in pb if x["conf"] is not None], atol=1e-12)
assert np.allclose(ln.get_ydata(), [x["acc"] for x in pb if x["acc"] is not None], atol=1e-12)
pdf, png = HERE / f"{NAME}.pdf", HERE / f"{NAME}.png"
fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
fig.savefig(png, dpi=300)
plt.close(fig)
print("gate 2: bars, curve and printed ECE/Brier from the JSON")

C.check_figure(NAME, FRAC, pdf)
C.summary()
