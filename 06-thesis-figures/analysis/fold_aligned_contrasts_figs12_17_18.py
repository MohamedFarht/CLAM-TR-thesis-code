#!/usr/bin/env python3
# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/fold_aligned_contrasts_figs12_17_18.py
# Thesis output : data of Figures 12, 17 and 18 (fold-aligned bestHP QWKs, contrast bootstrap intervals, projection) and the CI column of Table 11 (factorial_ci_table.csv); its own figures are earlier drafts; the printed figures come from plots/make_figs_12_17_18.py, which imports this file
# Inputs        : CLAM_TR_V20_DIR/{UNI,UNI2-h}/v20_final_summary.json (in data/) and per-slide prediction files predictions/{agg}_{hp}_fold{k}_predictions*.json (not in the repository); data/v16_validation/*.json; 03-optimization/CLAM_TR_v16_Validation_Experiments.ipynb
# Outputs       : staged/ next to this script (draft figures, figures_regen_results.json, factorial_ci_table.csv)
# Run           : python -P 06-thesis-figures/analysis/fold_aligned_contrasts_figs12_17_18.py
# Provenance    : adapted from the thesis figure scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Fold-aligned data of thesis Figures 12, 17 and 18 (and draft versions of the three figures).

Thesis figures (file names of the thesis build). The names inside this script (Fig. 18 / Fig. 19, draw_fig18,
draw_fig19, F18, F19) follow the figure numbers of an earlier draft: Fig. 18 = thesis Figure 17, Fig. 19 = thesis
Figure 18.
  Figure 12  fig_3_2_6_forest_plot.pdf     (fig:factorial_forest, Chapter 3)
  Figure 17  fig_4_X_multiseed_power.pdf   (fig:multiseed_power, Chapter 4)
  Figure 18  fig_4_Y_v16_vs_v20_power.pdf  (fig:v16_vs_v20_power, Chapter 4)

The first versions were drawn on Colab by a 42-cell copy of the April figures notebook (not in this repository):
cells 10 + 14 (Fig. 12), cell 16 (Fig. 18) and cell 17 (Fig. 19).
Those cells paired the per-fold QWK lists of `v20_final_summary.json` by list position, but the lists are stored
in completion order, which differs between configurations, so the paired differences compared different folds.

What this script does
  * Every stored fold QWK of the bestHP configurations is mapped to its fold index through the per-fold
    prediction files `{encoder}/predictions/{agg}_bestHP_fold{k}_predictions*.json` (QWK recomputed with
    quadratic weights over labels 0-5); exactly one file per fold must reproduce exactly one stored value
    (|diff| < 1e-6) and the matches must form a permutation. The validation slide set and the slide labels
    of each fold must be identical across all configurations and both encoders.
  * UNI2-h ABMIL_bestHP fold 0 uses the LOCAL prediction file, which is the original run (QWK 0.949984 =
    the value in v20_final_summary.json and the value behind every thesis number). The Drive copy of that
    file was overwritten by the 2026-05-02 retrain (QWK 0.947576) and is not used.
  * Fig. 12 plots the four contrasts that table `tab:factorial_decomp` defines, per encoder:
      Dual-branch         = DualBranch - ABMIL
      Heads and block     = RetNet - DualBranch      (earlier label "Enhancement block")
      Spatial-decay heads = SpatialMult - DualBranch (earlier label "Spatial decay")
      Interaction         = SpatialMultRetNet - RetNet - SpatialMult + DualBranch
    on fold-aligned bestHP differences, with the v20 notebook's paired bootstrap: a fresh
    np.random.RandomState(42) per contrast, 10,000 iterations of rng.choice(5, 5, replace=True), percentile
    95% interval (np.percentile [2.5, 97.5]). Row order follows the table (UNI block above UNI2-h).
    Nadeau-Bengio whiskers are NOT drawn by default (they are 2-3x wider than the bootstrap whiskers and
    would compress them; the caption and marker rule are bootstrap-based); `--nb-whiskers` adds them.
  * Fig. 18 keeps the notebook's normal-approximation projection (np.random.default_rng(42), 10,000 normal
    draws with SD = sd/sqrt(n) per n, percentile interval) on the fold-aligned UNI2-h
    SpatialMultRetNet_bestHP - ABMIL_bestHP differences.
  * Fig. 19: v20 bar = UNI SpatialMultRetNet_bestHP - ABMIL_bestHP (fold-aligned); v16 bar =
    V5_CONCAT_ONLY - V1_ABMIL from v16_all_results.json (its per-fold lists are in fold order: checked
    below against the per-fold printout of the v16 notebook). Both bars use the same v20 bootstrap as
    Fig. 12. Bars are labelled with the two different contrasts (they are not the same comparison).
  * Labels (labels only, no data change): Fig. 19 shows no run names ("600-slide development cohort" /
    "9,128-slide cohort"; "concatenation model without retention - non-gated ABMIL"); Fig. 12 row labels read
    "<encoder>: <contrast>" with a true minus sign (U+2212) in the definitions (CSV / JSON keep ASCII).
    The --old-titles variant is unchanged and still prints the original v16/v20 titles.

Style: the notebook's style cell (Cell 0) + setup cell (Cell 3: seaborn whitegrid + rcParams), with the
font pinned to DejaVu Sans, which is what Colab used for the originals (Arial is not installed there).
Only data loading/pairing, the Fig. 12 contrast definitions, and labels that were inaccurate change.

Outputs (written to the subfolder staged/ next to this script):
  fig_3_2_6_forest_plot.pdf/.png, fig_4_X_multiseed_power.pdf/.png, fig_4_Y_v16_vs_v20_power.pdf/.png
  (PNG previews at 200 dpi), figures_regen_results.json, factorial_ci_table.csv
The printed thesis figures are drawn from the same data by plots/make_figs_12_17_18.py, which imports this file.

Usage (no torch, no GPU, local files only):
  python -P fold_aligned_contrasts_figs12_17_18.py                  # default figures
  python -P fold_aligned_contrasts_figs12_17_18.py --nb-whiskers    # Fig. 12 with Nadeau-Bengio whiskers
  python -P fold_aligned_contrasts_figs12_17_18.py --old-titles     # keep the original Fig. 18/19 titles
"""
import argparse
import datetime
import glob
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))  # repo path (repository root)
BASE = os.environ.get("CLAM_TR_V20_DIR", os.path.join(HERE, "..", "data", "v20_fullscale"))  # repo path
V16_JSON = os.path.join(HERE, "..", "data", "v16_validation", "v16_all_results.json")  # repo path
V16_SUMMARY = os.path.join(HERE, "..", "data", "v16_validation", "v16_final_summary.json")  # repo path
V16_NB = os.path.join(ROOT, "03-optimization", "CLAM_TR_v16_Validation_Experiments.ipynb")  # repo path
OUT = os.path.join(HERE, "staged")

ENCS = ["UNI", "UNI2-h"]
AGGS = ["ABMIL", "DualBranch", "RetNet", "SpatialMult", "SpatialMultRetNet", "SpatialAdd"]
HP = "bestHP"
K = 5
N_CLASSES = 6
TOL = 1e-6
NB_FACTOR = 1.0 / 5 + 1.0 / 4          # Nadeau-Bengio: 1/J + n_test/n_train with 5-fold CV
COL_UNI, COL_UNI2H = "#648FFF", "#FE6100"
UNI2H_ABMIL_F0_ORIGINAL = 0.949984     # summary value = local original-run file
UNI2H_ABMIL_F0_RETRAINED = 0.947576    # Drive file after the 2026-05-02 retrain (not used)

for p in (BASE, V16_JSON, V16_SUMMARY, V16_NB):
    assert os.path.exists(p), f"missing input: {p}"
os.makedirs(OUT, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def qwk(y, p, n=N_CLASSES):
    """Quadratic-weighted Cohen kappa over labels 0..n-1 (full n x n matrix)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    assert y.shape == p.shape and y.min() >= 0 and p.min() >= 0 and y.max() < n and p.max() < n
    o = np.zeros((n, n), dtype=float)
    np.add.at(o, (y, p), 1.0)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    i, j = np.indices((n, n))
    w = (i - j) ** 2
    return 1.0 - (w * o).sum() / (w * e).sum()


# --------------------------------------------------------------------------------------- fold alignment
SUMMARY = {e: load_json(os.path.join(BASE, e, "v20_final_summary.json")) for e in ENCS}
FOLD_SLIDES = {}   # fold -> {slide_id: label}, must be identical for every configuration and encoder
CELLS = {}         # (enc, agg) -> dict


def align_cell(enc, agg, hp=HP):
    stored = [float(v) for v in SUMMARY[enc]["experiments"][f"{agg}_{hp}"]["fold_qwks"]]
    assert len(stored) == K, (enc, agg, stored)
    st = SUMMARY[enc]["experiments"][f"{agg}_{hp}"]
    assert abs(np.mean(stored) - st["mean_qwk"]) < 1e-9, (enc, agg)
    fold_rec = {}
    for k in range(K):
        cands = sorted(glob.glob(os.path.join(BASE, enc, "predictions", f"{agg}_{hp}_fold{k}_predictions*.json")))
        assert cands, f"no prediction file for {enc} {agg}_{hp} fold {k}"
        matched = []
        for c in cands:
            d = load_json(c)
            sl = [str(s) for s in d["slide_ids"]]
            y = np.asarray(d["labels"], dtype=int)
            pr = np.asarray(d["predictions"], dtype=int)
            assert len(sl) == len(y) == len(pr) and len(set(sl)) == len(sl), (c, len(sl))
            lab = dict(zip(sl, y.tolist()))
            if k in FOLD_SLIDES:
                assert FOLD_SLIDES[k] == lab, f"fold {k}: slide set or labels differ in {c}"
            else:
                FOLD_SLIDES[k] = lab
            q = qwk(y, pr)
            hits = [i for i, s in enumerate(stored) if abs(s - q) < TOL]
            if hits:
                matched.append((os.path.basename(c), q, hits))
        assert len(matched) == 1 and len(matched[0][2]) == 1, \
            f"{enc} {agg}_{hp} fold {k}: need exactly one file matching exactly one stored value, got {matched}"
        name, q, hits = matched[0]
        fold_rec[k] = {"file": name, "file_qwk": q, "stored_index": hits[0], "n_slides": len(FOLD_SLIDES[k])}
    idx = [fold_rec[k]["stored_index"] for k in range(K)]
    assert sorted(idx) == list(range(K)), f"{enc} {agg}: stored values do not map one-to-one onto folds: {idx}"
    completion_order = [None] * K           # completion_order[i] = fold id of stored list entry i
    for k in range(K):
        completion_order[fold_rec[k]["stored_index"]] = k
    q_fold = np.array([stored[fold_rec[k]["stored_index"]] for k in range(K)])
    for k in range(K):
        assert abs(q_fold[k] - fold_rec[k]["file_qwk"]) < TOL
    return {"stored_completion_order": stored, "fold_of_stored_entry": completion_order,
            "qwk_by_fold": q_fold, "files": fold_rec}


for e, a in itertools.product(ENCS, AGGS):
    CELLS[(e, a)] = align_cell(e, a)

# the thesis convention for UNI2-h ABMIL_bestHP fold 0 (original run)
_u2a = CELLS[("UNI2-h", "ABMIL")]
assert abs(_u2a["qwk_by_fold"][0] - UNI2H_ABMIL_F0_ORIGINAL) < TOL, _u2a["qwk_by_fold"][0]
assert _u2a["files"][0]["file"] == "ABMIL_bestHP_fold0_predictions.json"


def Q(enc, agg):
    return CELLS[(enc, agg)]["qwk_by_fold"]


# --------------------------------------------------------------------------------------- statistics
def boot_v20(diff, n_boot=10000, seed=42):
    """Exactly the v20 notebook's paired bootstrap (fresh RandomState per comparison)."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.RandomState(seed)
    bd = []
    for _ in range(n_boot):
        idx = rng.choice(len(diff), len(diff), replace=True)
        bd.append(diff[idx].mean())
    lo, hi = np.percentile(bd, [2.5, 97.5])
    return float(lo), float(hi)


def boot_notebook_default_rng(diff, n_boot=10000):
    """The thesis-figures notebook's bootstrap (Cells 10/17), reported for comparison only."""
    rng = np.random.default_rng(42)
    boots = rng.choice(np.asarray(diff, float), size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def describe(diff):
    d = np.asarray(diff, dtype=float)
    n = len(d)
    m, sd1 = float(d.mean()), float(d.std(ddof=1))
    tc = float(stats.t.ppf(0.975, n - 1))
    se_t, se_nb = sd1 / np.sqrt(n), sd1 * np.sqrt(NB_FACTOR)
    lo, hi = boot_v20(d)
    return {
        "fold_differences": [float(v) for v in d], "mean": m, "sd_ddof1": sd1, "sd_ddof0": float(d.std(ddof=0)),
        "boot95_v20": [lo, hi], "boot95_spans_zero": bool(lo <= 0 <= hi),
        "boot95_notebook_default_rng": list(boot_notebook_default_rng(d)),
        "t95": [m - tc * se_t, m + tc * se_t], "p_t": float(2 * stats.t.sf(abs(m / se_t), n - 1)),
        "nb95": [m - tc * se_nb, m + tc * se_nb], "p_nb": float(2 * stats.t.sf(abs(m / se_nb), n - 1)),
        "n_positive": int((d > 0).sum()), "n": n,
    }


# Fig. 12 contrasts (tab:factorial_decomp)
F12_DEFS = [
    ("Dual-branch", "DualBranch - ABMIL", lambda e: Q(e, "DualBranch") - Q(e, "ABMIL")),
    # rows named by what they add (labels only; data unchanged)
    ("Heads and block", "RetNet - DualBranch", lambda e: Q(e, "RetNet") - Q(e, "DualBranch")),
    ("Spatial-decay heads", "SpatialMult - DualBranch", lambda e: Q(e, "SpatialMult") - Q(e, "DualBranch")),
    ("Interaction", "(SpatialMultRetNet - SpatialMult)\n- (RetNet - DualBranch)",
     lambda e: Q(e, "SpatialMultRetNet") - Q(e, "RetNet") - Q(e, "SpatialMult") + Q(e, "DualBranch")),
]
F12 = []
for enc in ENCS:
    for name, label, fn in F12_DEFS:
        r = describe(fn(enc))
        r.update({"encoder": enc, "contrast": name, "definition": label.replace("\n", " ")})
        F12.append(r)

# Fig. 18
D18 = Q("UNI2-h", "SpatialMultRetNet") - Q("UNI2-h", "ABMIL")
F18 = describe(D18)
NS = np.array([5, 10, 25, 50, 100])
_rng18 = np.random.default_rng(42)
F18["projection"] = []
for n in NS:
    boots = _rng18.normal(F18["mean"], F18["sd_ddof1"] / np.sqrt(n), size=10000)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    z = float(stats.norm.ppf(0.975))
    F18["projection"].append({"n": int(n), "mc95": [float(lo), float(hi)],
                              "analytic_z95": [F18["mean"] - z * F18["sd_ddof1"] / np.sqrt(n),
                                               F18["mean"] + z * F18["sd_ddof1"] / np.sqrt(n)]})

# Fig. 19
V16 = load_json(V16_JSON)
V16S = load_json(V16_SUMMARY)["all_results"]
v16_v1 = np.array(V16["V1_ABMIL"]["qwk"]["values"], dtype=float)
v16_v5 = np.array(V16["V5_CONCAT_ONLY"]["qwk"]["values"], dtype=float)
assert np.allclose(v16_v1, V16S["V1_ABMIL"]["qwk_values"]) and np.allclose(v16_v5, V16S["V5_CONCAT_ONLY"]["qwk_values"])


def v16_fold_printout(nb_path):
    """Per-fold QWKs printed by run_5fold_cv in the v16 notebook, per experiment, in loop order."""
    nb = load_json(nb_path)
    text = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if "text" in o:
                text.append("".join(o["text"]) if isinstance(o["text"], list) else o["text"])
    text = "\n".join(text)
    assert "[Resume]" not in text and "SKIPPED" not in text, "v16 run was resumed: list order needs checking"
    out = {}
    for blk in re.split(r"EXPERIMENT: ", text)[1:]:
        exp = blk.split("\n", 1)[0].strip()
        out[exp] = [(int(f), float(q)) for f, q in re.findall(r"Fold (\d)/5\.\.\. QWK=([0-9.]+)", blk)]
    return out


_pr = v16_fold_printout(V16_NB)
for exp, vals in (("V1_ABMIL", v16_v1), ("V5_CONCAT_ONLY", v16_v5)):
    folds = [f for f, _ in _pr[exp]]
    assert folds == [1, 2, 3, 4, 5], (exp, folds)
    assert np.allclose([q for _, q in _pr[exp]], np.round(vals, 4), atol=5e-5), (exp, _pr[exp], vals)
F19 = {"v16": describe(v16_v5 - v16_v1), "v20": describe(Q("UNI", "SpatialMultRetNet") - Q("UNI", "ABMIL"))}
F19["v16"].update({"minuend": "V5_CONCAT_ONLY", "subtrahend": "V1_ABMIL",
                   "minuend_desc": V16["V5_CONCAT_ONLY"]["description"], "minuend_params": V16["V5_CONCAT_ONLY"]["params"],
                   "subtrahend_desc": V16["V1_ABMIL"]["description"], "subtrahend_params": V16["V1_ABMIL"]["params"],
                   "fold_order_check": "per-fold QWKs printed in loop order Fold 1/5..5/5 by the v16 notebook "
                                       "(single uninterrupted run, no resume) equal qwk.values in list order; same "
                                       "StratifiedKFold(5, shuffle=True, random_state=42) over the same 600 slides"})
F19["v20"].update({"minuend": "SpatialMultRetNet_bestHP", "subtrahend": "ABMIL_bestHP", "encoder": "UNI"})


# --------------------------------------------------------------------------------------- original (positional) values
def positional(enc, agg):
    return np.array(CELLS[(enc, agg)]["stored_completion_order"], dtype=float)


ORIG = {"fig12_cell10_pooled_contrasts_positional": [], "note": (
    "Values the ORIGINAL figures showed: per-fold lists paired by list position; Fig. 12 pooled the C1-C4 "
    "contrasts of Cell 10 with np.random.default_rng(42) (values rounded to 4 d.p. in factorial_ci_table.csv)")}
OLD_C = {"C1: retention vs no-retention": [("RetNet", "DualBranch"), ("SpatialMultRetNet", "SpatialMult")],
         "C2: spatial vs non-spatial": [("SpatialMult", "DualBranch"), ("SpatialMultRetNet", "RetNet")],
         "C3: spatial-add vs spatial-mult": [("SpatialAdd", "SpatialMult")],
         "C4: best vs ABMIL baseline": [("SpatialMultRetNet", "ABMIL"), ("RetNet", "ABMIL")]}
for enc in ENCS:
    for cname, pairs in OLD_C.items():
        d = np.concatenate([positional(enc, a) - positional(enc, b) for a, b in pairs])
        lo, hi = boot_notebook_default_rng(d)
        ORIG["fig12_cell10_pooled_contrasts_positional"].append(
            {"encoder": enc, "contrast": cname, "mean": round(float(d.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
             "spans_zero": bool(round(lo, 4) <= 0 <= round(hi, 4))})
_d = positional("UNI2-h", "SpatialMultRetNet") - positional("UNI2-h", "ABMIL")
_rng = np.random.default_rng(42)
ORIG["fig18_positional"] = {"mean": float(_d.mean()), "sd_ddof1": float(_d.std(ddof=1)), "projection": []}
for n in NS:
    lo, hi = np.percentile(_rng.normal(_d.mean(), _d.std(ddof=1) / np.sqrt(n), size=10000), [2.5, 97.5])
    ORIG["fig18_positional"]["projection"].append({"n": int(n), "mc95": [float(lo), float(hi)]})
_d = positional("UNI", "SpatialMultRetNet") - positional("UNI", "ABMIL")
ORIG["fig19_positional"] = {"v20_mean": float(_d.mean()), "v20_ci_default_rng": list(boot_notebook_default_rng(_d)),
                            "v16_mean": float((v16_v5 - v16_v1).mean()),
                            "v16_ci_default_rng": list(boot_notebook_default_rng(v16_v5 - v16_v1))}


# --------------------------------------------------------------------------------------- style
def apply_style():
    mpl.rcdefaults()
    # Cell 0 of the figures notebook (figure style)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["lines.linewidth"] = 1.5
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#009E73", "#999999", "#000000"]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    # Cell 3 (setup)
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                         "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
    # Colab had no Arial/Helvetica, so the originals were rendered in DejaVu Sans: pin it.
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["svg.hashsalt"] = "clam-tr"


def save(fig, name):
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, bbox_inches="tight", dpi=200)
    return pdf, png


def errorbar_rendered(cont):
    """Read back what an errorbar container actually draws: marker position and whisker ends."""
    line = cont.lines[0]
    x, y = (float(v) for v in line.get_xydata()[0])        # unit-converted (categorical x -> 0, 1)
    segs = cont.lines[2][0].get_segments()[0]
    return {"x": x, "y": y, "seg": [[float(segs[0][0]), float(segs[0][1])], [float(segs[1][0]), float(segs[1][1])]]}


# --------------------------------------------------------------------------------------- Fig. 12
MINUS = "−"


def true_minus(label):
    """Display form of an F12_DEFS definition: the ASCII operator ' - ' (and a line-initial '- ')
    becomes a true minus sign (U+2212). F12_DEFS itself stays ASCII for the CSV / JSON."""
    return "\n".join((MINUS + line[1:]) if line.startswith("- ") else line
                     for line in label.replace(" - ", f" {MINUS} ").split("\n"))


def draw_fig12(nb_whiskers=False):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    rows = F12                                   # table order, UNI first
    y = np.arange(len(rows))[::-1]
    labels = []
    rendered = []
    for i, r in enumerate(rows):
        color = COL_UNI if r["encoder"] == "UNI" else COL_UNI2H
        spans = r["boot95_spans_zero"]
        marker = "s" if spans else "o"
        lo, hi = r["boot95_v20"]
        if nb_whiskers:
            ax.plot(r["nb95"], [y[i], y[i]], color=color, linewidth=1.0, alpha=0.35, zorder=1, solid_capstyle="butt")
        cont = ax.errorbar(r["mean"], y[i], xerr=[[r["mean"] - lo], [hi - r["mean"]]],
                           fmt=marker, color=color, capsize=4, markersize=8,
                           markerfacecolor="white" if spans else color, markeredgewidth=1.5)
        rr = errorbar_rendered(cont)
        rendered.append({"encoder": r["encoder"], "contrast": r["contrast"], "marker_x": rr["x"],
                         "whisker": [rr["seg"][0][0], rr["seg"][1][0]], "marker": marker,
                         "filled": not spans})
        lab = dict((n, l) for n, l, _ in F12_DEFS)[r["contrast"]]
        labels.append(f"{r['encoder']}: {r['contrast']}\n{true_minus(lab)}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\Delta$ QWK (paired difference) with bootstrap 95% CI")
    ax.set_title("Factorial decomposition: per-encoder contrasts")
    ax.grid(axis="x", alpha=0.3)
    filled = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=7, label="CI excludes zero")
    hollow = mlines.Line2D([], [], color="black", marker="s", linestyle="None", markersize=7,
                           markerfacecolor="white", markeredgewidth=1.5, label="CI spans zero")
    handles = [filled, hollow]
    if nb_whiskers:
        handles.append(mlines.Line2D([], [], color="gray", alpha=0.5, linewidth=1.0,
                                     label="Nadeau-Bengio 95% CI"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    name = "fig_3_2_6_forest_plot" + ("_with_NB" if nb_whiskers else "")
    paths = save(fig, name)
    plt.close(fig)
    return paths, rendered


# --------------------------------------------------------------------------------------- Fig. 18
def draw_fig18(old_titles=False):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)
    mean = F18["mean"]
    rendered = []
    top = -np.inf
    for pr in F18["projection"]:
        n = pr["n"]
        lo, hi = pr["mc95"]
        color = COL_UNI2H if n == 5 else COL_UNI
        cont = ax.errorbar([n], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=4, markersize=8)
        rr = errorbar_rendered(cont)
        rendered.append({"n": n, "marker_y": rr["y"], "whisker": [rr["seg"][0][1], rr["seg"][1][1]]})
        ax.text(n, hi + 0.0008, f"n={n}", ha="center", fontsize=8)
        top = max(top, hi + 0.0008)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Effective sample size (5 folds $\times$ N seeds)")
    ax.set_ylabel("$\\Delta$ QWK\n(SpatialMultRetNet $-$ ABMIL on UNI2-h)")
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels(NS)
    lo_lim, hi_lim = ax.get_ylim()
    ax.set_ylim(lo_lim, max(hi_lim, top + 0.0022))      # head-room so the n=5 label sits inside the axes
    ax.set_title("Bootstrap CI projection vs effective sample size" if old_titles
                 else "Normal-approximation CI projection vs effective sample size")
    plt.tight_layout()
    paths = save(fig, "fig_4_X_multiseed_power")
    plt.close(fig)
    return paths, rendered


# --------------------------------------------------------------------------------------- Fig. 19
def draw_fig19(old_titles=False):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    rendered = {}
    # labels without run names: v16 -> "600-slide development cohort", v20 -> "9,128-slide cohort",
    # V5_CONCAT_ONLY -> "concatenation model without retention", V1_ABMIL -> "non-gated ABMIL" (the v16 V1
    # baseline has no gating; the v20 ABMIL is gated), matching the thesis caption.
    cat16 = "600-slide development cohort\nconcatenation model without retention\n$-$ non-gated ABMIL"
    cat20 = "9,128-slide cohort\nSpatialMultRetNet $-$ ABMIL"
    for key, cat, color, lab in (("v16", cat16, COL_UNI2H, "600-slide development cohort"),
                                 ("v20", cat20, COL_UNI, "9,128-slide cohort")):
        m = F19[key]["mean"]
        lo, hi = F19[key]["boot95_v20"]
        cont = ax.errorbar([cat], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=color, capsize=5, markersize=10,
                           label=lab)
        rr = errorbar_rendered(cont)
        rendered[key] = {"marker_y": rr["y"], "whisker": [rr["seg"][0][1], rr["seg"][1][1]]}
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.5, 1.5)
    if old_titles:
        ax.set_ylabel(r"$\Delta$ QWK (best aggregator $-$ ABMIL on UNI)")
        ax.set_title("Power gain from v16 (600) to v20 (9,128) cohorts")
    else:
        ax.set_ylabel(r"$\Delta$ QWK (model $-$ ABMIL baseline on UNI)")
        ax.set_title(r"Model $-$ ABMIL contrasts in the 600-slide and 9,128-slide cohorts")
    ax.legend(loc="best")
    plt.tight_layout()
    paths = save(fig, "fig_4_Y_v16_vs_v20_power")
    plt.close(fig)
    return paths, rendered


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-whiskers", action="store_true", help="Fig. 12 with Nadeau-Bengio whiskers (extra file)")
    ap.add_argument("--old-titles", action="store_true", help="keep the original Fig. 18/19 titles and y-label")
    args = ap.parse_args()

    p12, r12 = draw_fig12(nb_whiskers=False)
    extra = None
    if args.nb_whiskers:
        extra, _ = draw_fig12(nb_whiskers=True)
    p18, r18 = draw_fig18(old_titles=args.old_titles)
    p19, r19 = draw_fig19(old_titles=args.old_titles)

    # rendered == computed (the plotted numbers are the computed numbers)
    for row, rr in zip(F12, r12):
        assert abs(rr["marker_x"] - row["mean"]) < 1e-12
        assert np.allclose(rr["whisker"], row["boot95_v20"], atol=1e-12)
    for pr, rr in zip(F18["projection"], r18):
        assert abs(rr["marker_y"] - F18["mean"]) < 1e-12 and np.allclose(rr["whisker"], pr["mc95"], atol=1e-12)
    for key in ("v16", "v20"):
        assert abs(r19[key]["marker_y"] - F19[key]["mean"]) < 1e-12
        assert np.allclose(r19[key]["whisker"], F19[key]["boot95_v20"], atol=1e-12)

    # factorial_ci_table.csv (same role as the notebook's CSV, new contrasts)
    csv = os.path.join(OUT, "factorial_ci_table.csv")
    with open(csv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("Encoder,Contrast,Definition,Delta_QWK,SD_ddof1,CI95_lo,CI95_hi,CI_spans_zero\n")
        for r in F12:
            fh.write(f"{r['encoder']},{r['contrast']},\"{r['definition']}\",{r['mean']:.4f},{r['sd_ddof1']:.4f},"
                     f"{r['boot95_v20'][0]:.4f},{r['boot95_v20'][1]:.4f},{r['boot95_spans_zero']}\n")

    cells = {f"{e}/{a}_{HP}": {"qwk_by_fold": [float(v) for v in c["qwk_by_fold"]],
                               "stored_completion_order": c["stored_completion_order"],
                               "fold_of_stored_entry": c["fold_of_stored_entry"],
                               "files": {str(k): v for k, v in c["files"].items()}}
             for (e, a), c in CELLS.items()}
    res = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/"),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": mpl.__version__,
                     "seaborn": sns.__version__},
        "inputs": {**{f"v20_final_summary[{e}]": {"path": os.path.relpath(os.path.join(BASE, e, "v20_final_summary.json"), ROOT),
                                                 "sha256": sha256(os.path.join(BASE, e, "v20_final_summary.json"))} for e in ENCS},
                   "v16_all_results": {"path": os.path.relpath(V16_JSON, ROOT), "sha256": sha256(V16_JSON)}},
        "conventions": {
            "pairing": "fold index (stored completion-order lists mapped to folds via the per-fold prediction files)",
            "qwk": "quadratic-weighted Cohen kappa, labels 0-5",
            "bootstrap": "v20 notebook: fresh np.random.RandomState(42) per contrast, 10,000 x rng.choice(5, 5, replace=True), "
                         "np.percentile([2.5, 97.5])",
            "uni2h_abmil_bestHP_fold0": f"local original-run file, QWK {UNI2H_ABMIL_F0_ORIGINAL} (= v20_final_summary.json, "
                                        f"thesis); Drive file is the 2026-05-02 retrain, QWK {UNI2H_ABMIL_F0_RETRAINED}, not used",
            "nadeau_bengio": "t(4) with SE = sd * sqrt(1/5 + 1/4)",
            "slide_sets": {str(k): len(v) for k, v in FOLD_SLIDES.items()},
        },
        "cells_bestHP": cells,
        "fig12": {"file": os.path.basename(p12[0]), "rows": F12, "rendered": r12,
                  "nb_whiskers": "not drawn in the staged figure" + (f"; variant {os.path.basename(extra[0])}" if extra else "")},
        "fig18": {"file": os.path.basename(p18[0]), **F18, "rendered": r18},
        "fig19": {"file": os.path.basename(p19[0]), **F19, "rendered": r19},
        "original_figures_positional": ORIG,
    }
    with open(os.path.join(OUT, "figures_regen_results.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)

    print("Fig. 12 (bestHP, fold-aligned, v20 bootstrap):")
    for r in F12:
        print(f"  {r['encoder']:7s} {r['contrast']:18s} {r['mean']:+.4f} +/- {r['sd_ddof1']:.4f}  "
              f"boot [{r['boot95_v20'][0]:+.4f}, {r['boot95_v20'][1]:+.4f}] {'hollow' if r['boot95_spans_zero'] else 'filled'}  "
              f"NB [{r['nb95'][0]:+.4f}, {r['nb95'][1]:+.4f}]  diffs {np.round(r['fold_differences'], 4)}")
    print(f"Fig. 18: mean {F18['mean']:+.4f}, sd(ddof=1) {F18['sd_ddof1']:.4f}; p_t {F18['p_t']:.4f}, p_NB {F18['p_nb']:.4f}")
    for pr in F18["projection"]:
        print(f"  n={pr['n']:3d}: MC [{pr['mc95'][0]:+.4f}, {pr['mc95'][1]:+.4f}]  analytic z [{pr['analytic_z95'][0]:+.4f}, {pr['analytic_z95'][1]:+.4f}]")
    for key in ("v16", "v20"):
        r = F19[key]
        print(f"Fig. 19 {key}: {r['mean']:+.4f}  boot [{r['boot95_v20'][0]:+.4f}, {r['boot95_v20'][1]:+.4f}]  "
              f"(notebook default_rng [{r['boot95_notebook_default_rng'][0]:+.4f}, {r['boot95_notebook_default_rng'][1]:+.4f}])  "
              f"p_t {r['p_t']:.4f} p_NB {r['p_nb']:.4f}")
    print("Wrote:", *[os.path.relpath(p, HERE) for p in (*p12, *p18, *p19)], "staged/figures_regen_results.json",
          "staged/factorial_ci_table.csv", *([os.path.relpath(p, HERE) for p in extra] if extra else []))


if __name__ == "__main__":
    main()
