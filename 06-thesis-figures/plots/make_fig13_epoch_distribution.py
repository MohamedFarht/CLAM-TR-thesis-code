#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/plots/make_fig13_epoch_distribution.py
# Thesis output : Figure 13 (fig_2_X_epoch_distribution) and the best-epoch values of Table 19
# Inputs        : data/v20_fullscale/{UNI,UNI2-h}/v20_final_summary.json (best_epoch per fold); data/epoch_distribution_stats.csv; gates read CLAM_TR_PROJECT_DIR/thesis/main/bolumler/icerik/bolum_03_bulgular.tex; Each figure is compared at the end with the figure file of the thesis build (CLAM_TR_PROJECT_DIR/thesis/main/figures) by fig_checks.check_figure
# Outputs       : PDF + PNG next to this script
# Run           : python -P 06-thesis-figures/plots/make_fig13_epoch_distribution.py
# Provenance    : adapted from the thesis figure scripts; the drawing and the checks are unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Thesis Figure 13 (fig:train_dynamics, figure file fig_2_X_epoch_distribution.pdf) with every text item >= 10 pt
printed, Times New Roman. Writes fig_2_X_epoch_distribution.{pdf,png} next to this script.

Same data and the same gates as the generator of the previous version. Changes only:
  * canvas = printed size (1.0 \textwidth wide, scale 1.0; saved without a tight bounding box), so native = printed;
  * fonts Times New Roman (+ stix math), ticks / legend 10 pt, axis titles 10.5 pt, title 11 pt (previous version:
    Arial 7.2-8 pt printed at 0.85 \textwidth);
  * x tick labels rotated 25 degrees (previously 15) so that the longer names do not touch at 10 pt; legend moved
    right, above the DualBranch boxes (empty area).
Gates (asserted before anything is written): describe().round(1) of the data = data/epoch_distribution_stats.csv
(24 rows); per variant and encoder, mean and sample SD (ddof = 1) to 1 decimal = the Table 19 cells parsed from
the Chapter 3 LaTeX source of the thesis; tick order = Table 19 order; box quartiles = CSV; fig_checks (tokens,
sizes, overlap, fit).
Usage: python -P make_fig13_epoch_distribution.py
"""
import json
import os  # repo path (reads the environment variables of this repository)
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import cbook  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_checks as C  # noqa: E402  # repo path (module renamed)

NAME, FRAC = "fig_2_X_epoch_distribution", 1.0
RES = pathlib.Path(os.environ.get("CLAM_TR_V20_DIR", HERE.parent / "data" / "v20_fullscale"))  # repo path
STATS = HERE.parent / "data" / "epoch_distribution_stats.csv"  # repo path
TEX = C.ROOT / "thesis/main/bolumler/icerik/bolum_03_bulgular.tex"

# ---- Table 19 (tab:train_dynamics): row order and the epoch cells, parsed from the source
tex = TEX.read_text(encoding="utf-8")
m = re.search(r"\\label\{tab:train_dynamics\}(.*?)\\end\{tabular\}", tex, re.S)
assert m, "tab:train_dynamics not found"
rows = re.findall(r"\\texttt\{(\w+)\}\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*&\s*"
                  r"([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*\\\\", m.group(1))
assert len(rows) == 6, rows
ORDER = [r[0] for r in rows]
T19 = {r[0]: {"UNI": (float(r[3]), float(r[4])), "UNI2-h": (float(r[5]), float(r[6]))} for r in rows}
print("Table 19 order:", ORDER)

# ---- data
recs = []
for enc in ("UNI", "UNI2-h"):
    summary = json.load(open(RES / enc / "v20_final_summary.json", encoding="utf-8"))["experiments"]
    for key, exp in summary.items():
        if "_" not in key:
            continue
        agg, hp = key.rsplit("_", 1)
        for i, fm in enumerate(exp.get("fold_metrics", [])):
            recs.append({"encoder": enc, "aggregator": agg, "hp": hp, "fold": i, "best_epoch": int(fm.get("best_epoch", -1))})
df = pd.DataFrame(recs)
assert (df.best_epoch > 0).all()
got = df.groupby(["encoder", "aggregator", "hp"])["best_epoch"].describe().round(1).reset_index()
ref = pd.read_csv(STATS)
mm = ref.merge(got, on=["encoder", "aggregator", "hp"], suffixes=("_ref", "_got"))
assert len(ref) == len(got) == len(mm) == 24
for c in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
    assert (mm[c + "_ref"] - mm[c + "_got"]).abs().max() < 1e-9, c
print("gate 1: 24/24 rows identical to epoch_distribution_stats.csv")
best = df[df.hp == "bestHP"]
assert len(best) == 60 and set(best.aggregator) == set(ORDER)
for agg in ORDER:
    for enc in ("UNI", "UNI2-h"):
        v = best[(best.aggregator == agg) & (best.encoder == enc)].best_epoch
        assert len(v) == 5
        mean, sd = round(v.mean(), 1), round(v.std(ddof=1), 1)
        assert (mean, sd) == T19[agg][enc], (agg, enc, (mean, sd), T19[agg][enc])
        st = cbook.boxplot_stats(v.to_numpy())[0]
        r = ref[(ref.encoder == enc) & (ref.aggregator == agg) & (ref.hp == "bestHP")].iloc[0]
        assert round(st["med"], 1) == r["50%"] and round(st["q1"], 1) == r["25%"] and round(st["q3"], 1) == r["75%"]
print("gate 2: Table 19 epoch cells (mean +/- sample SD) = data; box quartiles = CSV, all 12 boxes")

# ---- figure (drawing of the previous version; fonts, canvas and rotation changed)
sns.set_style("whitegrid")
C.times_rc(plt)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "legend.title_fontsize": 10, "savefig.bbox": None})
fig, ax = plt.subplots(figsize=(FRAC * C.TW / 72, 3.7))
sns.boxplot(data=best, x="aggregator", y="best_epoch", hue="encoder", ax=ax, palette=["#648FFF", "#FE6100"],
            order=ORDER, hue_order=["UNI", "UNI2-h"])
ax.set_xlabel("Aggregator variant (bestHP)")
ax.set_ylabel("Best epoch")
ylim0 = ax.get_ylim()
ax.set_yticks(range(0, 40, 5))       # the old figure printed 0, 5, ..., 35
assert ax.get_ylim() == ylim0, (ax.get_ylim(), ylim0)
ax.set_title("Per-variant best-epoch distribution (5 folds per cell)")
plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
# legend above the DualBranch boxes (empty there, max 14); at the old upper-left place it sat just above the
# ABMIL outlier at 20
sns.move_legend(ax, "upper left", bbox_to_anchor=(0.175, 0.995))
fig.tight_layout(pad=0.3)
ticks = [t.get_text() for t in ax.get_xticklabels()]
assert ticks == ORDER, ticks
pdf, png = HERE / f"{NAME}.pdf", HERE / f"{NAME}.png"
fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
fig.savefig(png, dpi=300)
plt.close(fig)
print("x order = Table 19 order", ticks)

C.check_figure(NAME, FRAC, pdf)
C.summary()
