# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/lora_planned_tests.py
# Thesis output : Appendix Table 1 (planned Stage 4 tests of the LoRA protocol) and the TOST values of Appendix 1.4
# Inputs        : 05-experimental/lora_v1_results/*.json and the LoRA notebook (in the repository); CLAM_TR_V20_DIR/{UNI,UNI2-h}/v20_final_summary.json (in data/) and per-slide prediction files predictions/{agg}_{hp}_fold{k}_predictions*.json (not in the repository); consistency inputs under CLAM_TR_PROJECT_DIR: the thesis LaTeX sources of Chapters 3-4 and Appendix 1, the companion-paper file r2c1_c2_results.json and an earlier check log (not in the repository)
# Outputs       : lora_planned_tests_results.json, tab_lora_planned_tests.tex, REPORT_lora_tests.md next to this script
# Run           : python -P 06-thesis-figures/analysis/lora_planned_tests.py > lora_planned_tests_out.txt 2>&1
# Provenance    : adapted from the thesis analysis scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md. The checks that need inputs absent from this
#                 repository are skipped with a printed note (see "Optional inputs" below).
# ====================================================================================================
"""Planned Stage-4 tests of the pre-registered LoRA protocol (thesis Appendix 1, subsection app:lora_stats),
computed afterwards on the EXISTING per-fold results (Appendix Table 1).

No training, no GPU. Read-only; writes only into the folder of this script:
    lora_planned_tests_results.json   inputs (sha256), per-fold values, checks, test results
    tab_lora_planned_tests.tex        LaTeX table (label tab:lora_planned_tests)
    REPORT_lora_tests.md              report; every number in it is formatted from the computed values
Run:  python -P lora_planned_tests.py > lora_planned_tests_out.txt 2>&1

Fold pairing (per-fold scores are never paired by list position unless the list is provably in fold
order):
  * Frozen UNI fold QWKs are recomputed from the per-slide files {cfg}_fold{k}_predictions*.json (fold
    index in the file name) with sklearn cohen_kappa_score(weights="quadratic"). Each must reproduce a
    stored value of v20_final_summary.json (|diff| < 1e-9) and the five must reproduce the stored
    multiset; the stored lists are in completion order, not fold order.
  * LoRA Stage-4 fold QWKs have no per-slide file. The stored list is in fold order because
    run_stage_4() (notebook cell "def run_stage_4") builds it as
    [fold_qwks_dict[str(f)] for f in range(5)]; the line is asserted in the notebook source, and the
    logged "Fold k: QWK=" lines of the last Stage-4 run (SpatialAdd_bestHP) are asserted against it.
  * Fold slide sets: every frozen per-slide file must equal the LoRA Stage-2 held-out file of the same
    fold, slide by slide with identical labels (same kfold CSV via load_labels()); Stage 4 used the same
    load_labels() and logged "Available LoRA features: 9128", so no slide was dropped from its folds.
The script stops (exit 1) before computing any new test if a reproduction check fails.

Optional inputs (not in this repository). The checks that read them are skipped, with a printed note, when the
files are absent; no computed value depends on them:
  * the thesis LaTeX sources under CLAM_TR_PROJECT_DIR/thesis/main/bolumler/: the computed values are looked up in
    the printed sentences and tables. Some sentences were reworded after this script was written (the numbers did
    not change), so a few of these checks fail against the final thesis text (TABLES_AND_FIGURES.md, note A);
  * CLAM_TR_FOLD_QWK_COPY (default data/r2c1_c2_results.json): an independent copy of the fold-order QWKs,
    computed from the Drive copies of the prediction files (the file is in the sibling repository CLAM-TR);
  * CLAM_TR_LORA_CHECK_LOG (default data/lora_checks_out.txt): the log of an earlier check of the LoRA results.
"""
import hashlib
import itertools
import json
import os  # repo path (reads the environment variables of this repository)
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
import statsmodels
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CLAM_TR_PROJECT_DIR", Path(__file__).resolve().parents[2]))  # repo path
LORA_DIR = Path(__file__).resolve().parents[2] / "05-experimental" / "lora_v1_results"  # repo path
LORA_NB = Path(__file__).resolve().parents[2] / "05-experimental" / "CLAM_TR_LoRA_v1_FIXED_v8_SHARDED.ipynb"  # repo path
FROZEN = Path(os.environ.get("CLAM_TR_V20_DIR", Path(__file__).resolve().parents[1] / "data" / "v20_fullscale"))  # repo path
CH3 = ROOT / "thesis" / "main" / "bolumler" / "icerik" / "bolum_03_bulgular.tex"
CH4 = ROOT / "thesis" / "main" / "bolumler" / "icerik" / "bolum_04_tartisma.tex"
APP = ROOT / "thesis" / "main" / "bolumler" / "son_kisim" / "bolum_ekler.tex"
R2C = Path(os.environ.get("CLAM_TR_FOLD_QWK_COPY", Path(__file__).resolve().parents[1] / "data" / "r2c1_c2_results.json"))  # repo path (optional input, not in the repository)
J3CHK = Path(os.environ.get("CLAM_TR_LORA_CHECK_LOG", Path(__file__).resolve().parents[1] / "data" / "lora_checks_out.txt"))  # repo path (optional input, not in the repository)
OUT_JSON = HERE / "lora_planned_tests_results.json"
OUT_TEX = HERE / "tab_lora_planned_tests.tex"
OUT_MD = HERE / "REPORT_lora_tests.md"
for _p in (LORA_DIR, LORA_NB, FROZEN):
    assert _p.exists(), f"missing input: {_p}"
# optional inputs (not in this repository): the checks that read them are skipped, with a printed note, if absent
HAVE_THESIS = all(_p.exists() for _p in (CH3, CH4, APP))
HAVE_QWK_COPY = R2C.exists()
HAVE_CHECK_LOG = J3CHK.exists()
assert HERE.name == "analysis" and HERE.parent.name == "06-thesis-figures", HERE  # repo path

AGGS = ["ABMIL", "DualBranch", "SpatialMult", "SpatialAdd", "RetNet", "SpatialMultRetNet"]
HPS = ["bestHP", "defaultHP"]
CFGS = [f"{a}_{h}" for h in HPS for a in AGGS]
K = 5
N_SLIDES = 9128
NB_FACTOR = 1 / 5 + 1 / 4      # thesis Ch.2 sec:stat_plan: variance s^2/5 -> (1/5 + 1/4) s^2, four df
B_BOOT, BOOT_SEED = 10000, 42  # thesis Ch.2 sec:bootstrap: B = 10,000, percentile method
ALPHA = 0.05
# the eleven pre-specified in-encoder contrasts of thesis Ch.2 sec:stat_scope, (a)-(k)
ELEVEN = [("a", "RetNet", "ABMIL", "bestHP"), ("b", "DualBranch", "ABMIL", "bestHP"),
          ("c", "RetNet", "DualBranch", "bestHP"), ("d", "SpatialMult", "DualBranch", "bestHP"),
          ("e", "SpatialMult", "RetNet", "bestHP"), ("f", "SpatialMultRetNet", "RetNet", "bestHP"),
          ("g", "SpatialMultRetNet", "SpatialMult", "bestHP"), ("h", "SpatialAdd", "SpatialMult", "bestHP"),
          ("i", "RetNet", "ABMIL", "defaultHP"), ("j", "SpatialMult", "DualBranch", "defaultHP"),
          ("k", "SpatialAdd", "SpatialMult", "defaultHP")]
KEYS = [(a, b, h) for _, a, b, h in ELEVEN]
I_C1 = KEYS.index(("RetNet", "ABMIL", "bestHP"))

CHECKS = []


def check(name, ok, detail=""):
    ok = bool(ok)
    CHECKS.append({"check": name, "pass": ok, "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_pred(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    ids = list(d["slide_ids"])
    pr = np.asarray(d["predictions"], int)
    lb = np.asarray(d["labels"], int)
    assert len(ids) == len(pr) == len(lb) == len(set(ids)), p
    return ids, pr, lb


def qwk(lb, pr):
    assert {int(v) for v in np.unique(lb)} == set(range(6)), "a grade is missing from the fold labels"
    return float(cohen_kappa_score(lb, pr, weights="quadratic"))


def boot_ci(d):
    """Percentile bootstrap of the mean fold difference: folds resampled with replacement, B = 10,000,
    numpy RandomState(42) re-seeded for every comparison (the convention of the thesis scripts)."""
    rng = np.random.RandomState(BOOT_SEED)
    n = len(d)
    b = [d[rng.choice(n, n, replace=True)].mean() for _ in range(B_BOOT)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def boot_ci_exact(d):
    """Same percentile interval over all 5^5 equally likely ordered resamples (seed-free cross-check)."""
    idx = np.array(list(itertools.product(range(len(d)), repeat=len(d))))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


FOLD_RATIO = None  # mean n_test/n_train over the five folds of this split, set in section 2


def paired(x, y=None):
    """Fold-paired statistics of d = x - y (or of x itself when y is None); x, y in fold order 0-4."""
    x = np.asarray(x, float)
    if y is None:
        d = x
        p_t = float(stats.ttest_1samp(d, 0.0).pvalue)
    else:
        y = np.asarray(y, float)
        d = x - y
        p_t = float(stats.ttest_rel(x, y).pvalue)
        assert abs(p_t - float(stats.ttest_1samp(d, 0.0).pvalue)) < 1e-12
    n = len(d)
    assert n == K and np.all(d != 0.0)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    p_w = float(stats.wilcoxon(d, method="exact").pvalue)  # exact, two-sided
    tq = float(stats.t.ppf(0.975, n - 1))
    se_nb = sd * np.sqrt(NB_FACTOR)
    p_nb = float(2 * stats.t.sf(abs(m / se_nb), n - 1))
    se_fs = sd * np.sqrt(1 / K + FOLD_RATIO)
    p_nb_fs = float(2 * stats.t.sf(abs(m / se_fs), n - 1))
    lo, hi = boot_ci(d)
    elo, ehi = boot_ci_exact(d)
    return {"diffs_fold0to4": d.tolist(), "mean": m, "sd_ddof1": sd, "n_pos": int((d > 0).sum()),
            "p_t": p_t, "p_W_exact": p_w, "d_z": m / sd, "boot_ci95": [lo, hi],
            "boot_ci95_exact_enumeration": [elo, ehi], "p_NB": p_nb,
            "ci95_NB": [m - tq * se_nb, m + tq * se_nb], "ci95_t": [m - tq * sd / np.sqrt(n), m + tq * sd / np.sqrt(n)],
            "p_NB_foldsize_ratio": p_nb_fs}


def bh(p):
    return [float(q) for q in multipletests(np.asarray(p, float), alpha=ALPHA, method="fdr_bh")[1]]


def bh_manual(p):
    """Benjamini & Hochberg (1995) step-up q-values with monotonisation (minimum from the right)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out.tolist()


def norm(s):
    return re.sub(r"\s+", " ", s)


print("lora_planned_tests --", datetime.now().isoformat(timespec="seconds"))
print(f"python {sys.version.split()[0]}  numpy {np.__version__}  scipy {scipy.__version__}  "
      f"sklearn {sklearn.__version__}  statsmodels {statsmodels.__version__}")

# ------------------------------------------------------------------ 1. input integrity
print("\n== 1. Input integrity (LoRA JSONs vs the sha256 list in lora_v1_results/README.md)")
readme = (LORA_DIR / "README.md").read_text(encoding="utf-8")
REF_SHA = {m.group(2): m.group(1) for m in re.finditer(r"^([0-9a-f]{64})\s+(\S+)\s*$", readme, re.M)}
check("README lists 9 sha256 hashes", len(REF_SHA) == 9, ", ".join(sorted(REF_SHA)))
INPUT_SHA = {}
for nm in ["v20_lora_features_results.json", "lora_training_results.json"] + [f"lora_fold{k}_predictions.json" for k in range(K)]:
    h = sha256(LORA_DIR / nm)
    INPUT_SHA[str(LORA_DIR / nm)] = h
    check(f"sha256 {nm} == README", h == REF_SHA.get(nm), h[:12])

# ------------------------------------------------------------------ 2. fold slide sets
print("\n== 2. Fold slide sets (LoRA Stage-2 held-out files = the kfold CSV split of load_labels())")
TR = json.loads((LORA_DIR / "lora_training_results.json").read_text(encoding="utf-8"))
S2 = {int(fm["fold"]): fm for fm in TR["fold_metrics"]}
check("Stage-2 fold_metrics carry the explicit folds 0-4", sorted(S2) == list(range(K)))
FOLD_IDS, FOLD_LAB, S2_QWK = {}, {}, {}
for k in range(K):
    ids, pr, lb = load_pred(LORA_DIR / f"lora_fold{k}_predictions.json")
    FOLD_IDS[k] = frozenset(ids)
    FOLD_LAB[k] = dict(zip(ids, lb.tolist()))
    S2_QWK[k] = qwk(lb, pr)
    check(f"lora_fold{k}_predictions.json reproduces Stage-2 fold_metrics[fold={k}].qwk",
          abs(S2_QWK[k] - float(S2[k]["qwk"])) < 1e-9, f"{S2_QWK[k]:.6f} vs {float(S2[k]['qwk']):.6f}")
FOLD_N = [len(FOLD_IDS[k]) for k in range(K)]
check("fold sets disjoint, union = 9,128 slides", sum(FOLD_N) == len(frozenset().union(*FOLD_IDS.values())) == N_SLIDES,
      str(FOLD_N))
FOLD_RATIO = float(np.mean([n / (N_SLIDES - n) for n in FOLD_N]))
print(f"  fold sizes {FOLD_N}; mean n_test/n_train = {FOLD_RATIO:.5f} (the thesis definition uses 1/4)")

# ------------------------------------------------------------------ 3. LoRA Stage 4
print("\n== 3. LoRA Stage-4 fold QWKs (v20_lora_features_results.json) and the proof of their fold order")
S4 = json.loads((LORA_DIR / "v20_lora_features_results.json").read_text(encoding="utf-8"))
check("Stage-4 file holds the 12 configurations", sorted(S4) == sorted(CFGS))
LORA = {}
for cfg in CFGS:
    a = np.asarray(S4[cfg]["fold_qwks"], float)
    LORA[cfg] = a
    check(f"{cfg}: 5 values; stored mean and population std reproduced",
          len(a) == K and abs(a.mean() - S4[cfg]["mean_qwk"]) < 1e-12 and abs(a.std() - S4[cfg]["std_qwk"]) < 1e-12)
NB = json.loads(LORA_NB.read_text(encoding="utf-8"))
C4 = [c for c in NB["cells"] if c["cell_type"] == "code" and "def run_stage_4" in "".join(c["source"])]
check("notebook has exactly one run_stage_4 cell", len(C4) == 1)
SRC4 = "".join(C4[0]["source"])
for needle in ("fold_qwks = [fold_qwks_dict[str(f)] for f in range(5)]",
               "fold_qwks_dict[str(fold)] = best_qwk_f",
               "val_df = df[df['kfold'] == fold].reset_index(drop=True)",
               "df = load_labels()"):
    check(f"run_stage_4 source contains: {needle}", needle in SRC4)
OUT4 = "".join("".join(o.get("text", "")) for o in C4[0].get("outputs", []))
check("Stage-4 log: 'Available LoRA features: 9128'", "Available LoRA features: 9128" in OUT4)
seg = OUT4[OUT4.index("[12/12] SpatialAdd_bestHP"):OUT4.index("SpatialAdd_bestHP: QWK =")]
LOGGED = [(int(f), float(q)) for f, q in re.findall(r"Fold (\d): QWK=([0-9.]+)", seg)]
check("Stage-4 log: 'Fold k: QWK=' lines of the last run (SpatialAdd_bestHP) are folds 0-4 in order",
      [f for f, _ in LOGGED] == list(range(K)), str(LOGGED))
check("logged fold QWKs == stored list at the same fold index (4 dp)",
      len(LOGGED) == K and all(f"{q:.4f}" == f"{LORA['SpatialAdd_bestHP'][f]:.4f}" for f, q in LOGGED))
summ_tab = OUT4[OUT4.index("STAGE 4 COMPLETE"):]
for cfg in CFGS:
    m = re.search(r"^" + re.escape(cfg) + r"\s+([0-9.]+) \+/- ([0-9.]+)\s*$", summ_tab, re.M)
    check(f"Stage-4 log summary line {cfg} == stored mean / population std",
          m is not None and (m.group(1), m.group(2)) == (f"{LORA[cfg].mean():.4f}", f"{LORA[cfg].std():.4f}"))

# ------------------------------------------------------------------ 4. frozen fold QWKs
print("\n== 4. Frozen fold QWKs recomputed from the per-slide prediction files (fold index in the file name)")
SUMM = {e: json.loads((FROZEN / e / "v20_final_summary.json").read_text(encoding="utf-8")) for e in ("UNI", "UNI2-h")}
FROZEN_FILES = {}
DUP_NOTES = {}


def frozen(enc, cfg):
    stored = [float(v) for v in SUMM[enc]["experiments"][cfg]["fold_qwks"]]
    vals, pos, notes = [], [], []
    ok_file, ok_set = True, True
    for k in range(K):
        cands = sorted((FROZEN / enc / "predictions").glob(f"{cfg}_fold{k}_predictions*.json"))
        assert cands, f"no prediction file for {enc} {cfg} fold {k}"
        rows = []
        for c in cands:
            ids, pr, lb = load_pred(c)
            q = qwk(lb, pr)
            rows.append({"file": c, "q": q, "ids": ids, "lab": lb,
                         "match": [i for i, s in enumerate(stored) if abs(q - s) < 1e-9]})
        good = [r for r in rows if r["match"]]
        ok_file &= len(good) >= 1 and len({round(r["q"], 12) for r in good}) == 1
        if len(cands) > 1:
            notes.append(f"fold {k}: " + "; ".join(f"{r['file'].name} QWK {r['q']:.6f}"
                                                  f" {'matches' if r['match'] else 'matches no stored value'}" for r in rows))
        r = good[0] if good else rows[0]
        same = frozenset(r["ids"]) == FOLD_IDS[k] and all(FOLD_LAB[k][s] == int(l) for s, l in zip(r["ids"], r["lab"]))
        ok_set &= same
        vals.append(r["q"])
        pos.append(r["match"][0] if r["match"] else None)
        FROZEN_FILES[f"{enc}/{cfg}/fold{k}"] = {"file": str(r["file"]), "sha256": sha256(r["file"])}
    check(f"{enc} {cfg}: each fold's per-slide file reproduces a stored fold QWK (|diff| < 1e-9)", ok_file,
          " | ".join(notes))
    if notes:
        DUP_NOTES[f"{enc} {cfg}"] = notes
    check(f"{enc} {cfg}: fold k slide set and labels == LoRA fold k for k = 0-4", ok_set)
    check(f"{enc} {cfg}: the five per-file QWKs == stored multiset; stored-list position of folds 0-4 = {pos}",
          np.allclose(sorted(vals), sorted(stored), rtol=0, atol=1e-9))
    return np.asarray(vals), pos


FROZ, FROZ_POS = {}, {}
for cfg in CFGS:
    FROZ[cfg], FROZ_POS[cfg] = frozen("UNI", cfg)
U2_SMR, U2_SMR_POS = frozen("UNI2-h", "SpatialMultRetNet_bestHP")
if HAVE_QWK_COPY:
    R2 = json.loads(R2C.read_text(encoding="utf-8"))
    n_r2 = 0
    for pc in R2["per_config"]:
        if pc["encoder"] == "UNI" or (pc["encoder"] == "UNI2-h" and pc["aggregator"] == "SpatialMultRetNet"):
            mine = FROZ[pc["aggregator"] + "_bestHP"] if pc["encoder"] == "UNI" else U2_SMR
            n_r2 += 1
            check(f"independent copy r2c1_c2_results.json (Drive files, fold order, 6 dp) == local: {pc['encoder']} {pc['aggregator']}_bestHP",
                  all(f"{a:.6f}" == f"{b:.6f}" for a, b in zip(pc["fold_qwk_in_fold_order"], mine)))
    check("r2c1_c2 cross-check covered 7 configurations", n_r2 == 7)
else:
    print(f"  note: {R2C.name} (independent copy of the fold-order QWKs; set CLAM_TR_FOLD_QWK_COPY) not found: "
          "this cross-check is skipped")

# ------------------------------------------------------------------ 5. thesis reproduction
print("\n== 5. Reproduction of the numbers printed in the thesis (computed value formatted -> found in the .tex)")
if HAVE_THESIS:
    SRC = {"Ch3": norm(CH3.read_text(encoding="utf-8")), "Ch4": norm(CH4.read_text(encoding="utf-8")),
           "App1": norm(APP.read_text(encoding="utf-8"))}
else:
    SRC = None
    print("  note: the thesis LaTeX sources were not found under CLAM_TR_PROJECT_DIR (they are not part of this "
          "repository): the checks against the printed thesis text are skipped")


def printed(label, s, where):
    if SRC is None:  # thesis sources not available: check skipped
        return
    for w in where:
        check(f"{label}: '{s}' printed in {w}", s in SRC[w])


def mpm(a):
    return "$" + f"{a.mean():.4f}" + r" \pm " + f"{a.std(ddof=1):.4f}" + "$"


if SRC is not None:
    t9 = SRC["Ch3"][SRC["Ch3"].index(r"\label{tab:main_factorial}"):]
    t9 = t9[:t9.index(r"\end{tabular}")]
    T9_UNI, T9_UNI2 = t9[:t9.index("UNI2-h encoder")], t9[t9.index("UNI2-h encoder"):]


def t9cell(block, cfg):
    i = block.index(r"\texttt{" + cfg.replace("_", r"\_") + "}")
    mm = re.search(r"([01]\.[0-9]{4}) \$\\pm\$ ([01]\.[0-9]{4})", block[i:i + 160])
    return mm.group(1), mm.group(2)


if SRC is not None:
    for a in AGGS:
        cfg = f"{a}_bestHP"
        got, exp = (f"{FROZ[cfg].mean():.4f}", f"{FROZ[cfg].std(ddof=1):.4f}"), t9cell(T9_UNI, cfg)
        check(f"Table tab:main_factorial UNI {cfg} mean +/- sd(ddof=1)", got == exp, f"computed {got}, printed {exp}")
    got, exp = (f"{U2_SMR.mean():.4f}", f"{U2_SMR.std(ddof=1):.4f}"), t9cell(T9_UNI2, "SpatialMultRetNet_bestHP")
    check("Table tab:main_factorial UNI2-h SpatialMultRetNet_bestHP mean +/- sd(ddof=1)", got == exp, f"computed {got}, printed {exp}")

# Table tab:decomp_stats_uni (eleven frozen contrasts): validates every test implementation used below
FRO11 = {k: paired(FROZ[f"{k[0]}_{k[2]}"], FROZ[f"{k[1]}_{k[2]}"]) for k in KEYS}
FQ_T = bh([FRO11[k]["p_t"] for k in KEYS])
FQ_NB = bh([FRO11[k]["p_NB"] for k in KEYS])
check("BH: statsmodels fdr_bh == manual step-up (frozen family, on p_t and on p_NB)",
      np.allclose(FQ_T, bh_manual([FRO11[k]["p_t"] for k in KEYS]), atol=1e-15)
      and np.allclose(FQ_NB, bh_manual([FRO11[k]["p_NB"] for k in KEYS]), atol=1e-15))
if SRC is not None:
    t12 = SRC["Ch3"][SRC["Ch3"].index(r"\label{tab:decomp_stats_uni}"):]
    t12 = t12[:t12.index(r"\end{tabular}")]
    ROWS12 = re.findall(r"(\w+) vs (\w+) \((bestHP|defaultHP)\) & \$([+-][0-9.]+)\$ & ([0-9.]+) & ([0-9.]+) & "
                        r"\$([+-][0-9.]+)\$ & ([0-9.]+) &", t12)
    check("Table tab:decomp_stats_uni: eleven rows parsed", len(ROWS12) == 11 and {(r[0], r[1], r[2]) for r in ROWS12} == set(KEYS))
    for a, b, h, dm, pt, pw, dd, q in ROWS12:
        i = KEYS.index((a, b, h))
        r = FRO11[(a, b, h)]
        got = (f"{r['mean']:+.5f}", f"{r['p_t']:.4f}", f"{r['p_W_exact']:.4f}", f"{r['d_z']:+.3f}", f"{FQ_T[i]:.4f}")
        check(f"Table tab:decomp_stats_uni row {a} vs {b} ({h}): dQWK, p_t, p_W, d, q_BH", got == (dm, pt, pw, dd, q),
              f"computed {got}")
r = FRO11[KEYS[I_C1]]
printed("frozen C1 Nadeau-Bengio p and BH q", r"the first row gives $p=" + f"{r['p_NB']:.4f}" + r"$ and $q_{\mathrm{BH}}="
        + f"{FQ_NB[I_C1]:.4f}" + "$", ["Ch3"])
printed("frozen C1 Nadeau-Bengio 95% CI", r"$p=" + f"{r['p_NB']:.4f}" + r"$ (95\% CI $[" + f"{r['ci95_NB'][0]:+.4f}, "
        + f"{r['ci95_NB'][1]:+.4f}" + "]$)", ["Ch3"])
rc, rd = FRO11[("RetNet", "DualBranch", "bestHP")], FRO11[("SpatialMult", "DualBranch", "bestHP")]
printed("bootstrap + NB CI of RetNet - DualBranch (bestHP)", r"difference is $[" + f"{rc['boot_ci95'][0]:+.4f}, {rc['boot_ci95'][1]:+.4f}"
        + r"]$, whereas the corresponding Nadeau--Bengio interval $[" + f"{rc['ci95_NB'][0]:+.4f}, {rc['ci95_NB'][1]:+.4f}" + "]$",
        ["Ch3"])
printed("bootstrap CI of SpatialMult - DualBranch (bestHP)", r"has CI $[" + f"{rc['boot_ci95'][0]:+.4f}, {rc['boot_ci95'][1]:+.4f}"
        + r"]$ and \texttt{SpatialMult\_bestHP} minus \texttt{DualBranch\_bestHP} has CI $["
        + f"{rd['boot_ci95'][0]:+.4f}, {rd['boot_ci95'][1]:+.4f}" + "]$", ["Ch3"])

# Appendix 1.4 / Ch.4 LoRA numbers
printed("best LoRA configuration SpatialAdd_defaultHP", mpm(LORA["SpatialAdd_defaultHP"]), ["App1", "Ch4"])
check("SpatialAdd_defaultHP is the best of the 12 LoRA configurations",
      max(CFGS, key=lambda c: LORA[c].mean()) == "SpatialAdd_defaultHP")
S2A = np.array([float(S2[k]["qwk"]) for k in range(K)])
printed("Stage-2 direct classifier", mpm(S2A), ["App1"])
printed("decision-rule baseline UNI2-h SpatialMultRetNet_bestHP", mpm(U2_SMR), ["App1", "Ch4"])
C1F = FROZ["RetNet_bestHP"] - FROZ["ABMIL_bestHP"]
C1L = LORA["RetNet_bestHP"] - LORA["ABMIL_bestHP"]
RED = 1 - C1L.mean() / C1F.mean()
printed("Contrast 1 frozen and LoRA", "shifts from $" + f"{C1F.mean():+.4f}" + "$ on the frozen encoder to $"
        + f"{C1L.mean():+.4f}" + "$ on the", ["App1", "Ch4"])
printed("Contrast 1 reduction", r"approximately~$" + f"{100 * RED:.0f}" + r"\%$", ["App1", "Ch4"])
RANK_F = sorted(AGGS, key=lambda a: -FROZ[f"{a}_bestHP"].mean())
RANK_L = sorted(AGGS, key=lambda a: -LORA[f"{a}_bestHP"].mean())
check("top bestHP slot RetNet on both feature sets; SpatialMult second on LoRA",
      RANK_F[0] == RANK_L[0] == "RetNet" and RANK_L[1] == "SpatialMult", f"frozen {RANK_F} | LoRA {RANK_L}")
printed("top slot", r"(\texttt{RetNet\_bestHP}, $" + f"{LORA['RetNet_bestHP'].mean():.4f}"
        + r"$, just ahead of \texttt{SpatialMult\_bestHP} at $" + f"{LORA['SpatialMult_bestHP'].mean():.4f}" + "$)", ["App1", "Ch4"])
TAU = float(stats.kendalltau([FROZ[f"{a}_bestHP"].mean() for a in AGGS], [LORA[f"{a}_bestHP"].mean() for a in AGGS]).statistic)
printed("Kendall tau of the six bestHP ranks", r"Kendall $\tau = " + f"{TAU:.2f}" + "$", ["App1", "Ch4"])
D6 = {a: float((LORA[f"{a}_bestHP"] - FROZ[f"{a}_bestHP"]).mean()) for a in AGGS}
check("six paired differences inside +/-0.012; largest negative RetNet, largest positive SpatialMult",
      max(abs(v) for v in D6.values()) < 0.012 and min(D6, key=D6.get) == "RetNet" and max(D6, key=D6.get) == "SpatialMult")
printed("largest negative shift", r"\texttt{RetNet} ($" + f"{D6['RetNet']:+.4f}" + "$)", ["App1", "Ch4"])
printed("largest positive shift", r"\texttt{SpatialMult} ($" + f"{D6['SpatialMult']:+.4f}" + "$)", ["App1", "Ch4"])
printed("range of the six differences", "range from $" + f"{D6['RetNet']:+.4f}" + "$ to $" + f"{D6['SpatialMult']:+.4f}" + "$", ["Ch4"])
printed("bound of the six differences", r"$\pm 0.012$", ["App1", "Ch4"])
LM = [LORA[c].mean() for c in CFGS]
printed("LoRA band", "[$" + f"{min(LM):.4f}" + "$, $" + f"{max(LM):.4f}" + "$]", ["App1"])
printed("LoRA band width", "$" + f"{max(LM) - min(LM):.3f}" + "$ QWK band", ["App1"])
DT = LORA["SpatialAdd_defaultHP"] - U2_SMR
P_LO = stats.ttest_1samp(DT, -0.005, alternative="greater").pvalue
P_HI = stats.ttest_1samp(DT, 0.005, alternative="less").pvalue
SE_T = DT.std(ddof=1) * np.sqrt(NB_FACTOR)
P_LO_NB, P_HI_NB = stats.t.sf((DT.mean() + 0.005) / SE_T, K - 1), stats.t.cdf((DT.mean() - 0.005) / SE_T, K - 1)
RULE_QUOTE = (r"A sign-flip on either contrast, or a magnitude reduction by more than half of the original effect, would be "
              r"evidence that the frozen-encoder result is encoder-state-specific and would limit the generalisation claim of "
              r"Contribution~3.")
printed("pre-registered contrast rule quoted verbatim", RULE_QUOTE, ["App1"])
printed("TOST paired difference", "paired difference of $" + f"{DT.mean():+.4f}" + "$ QWK", ["App1"])
printed("TOST naive p", r"(\textit{p}~$= " + f"{max(P_LO, P_HI):.3f}" + "$)", ["App1", "Ch4"])
printed("TOST Nadeau-Bengio p", r"(\textit{p}~$= " + f"{max(P_LO_NB, P_HI_NB):.3f}" + "$)", ["App1", "Ch4"])

# ------------------------------------------------------------------ 6. earlier check of the LoRA results (optional input)
print("\n== 6. Reproduction of an earlier check of the LoRA results (lora_checks_out.txt)")
if HAVE_CHECK_LOG:
    JT = J3CHK.read_text(encoding="utf-8")
    n_j = 0
    for m in re.finditer(r"^\s+(\w+_(?:bestHP|defaultHP))\s+LoRA mean=([0-9.]+) sd1=([0-9.]+) \| frozen mean=([0-9.]+) \| "
                         r"delta\(mean diff\)=([+-][0-9.]+) fold-paired sd1=([0-9.]+) pos=(\d)/5 p_t=([0-9.]+) p_NB=([0-9.]+)",
                         JT, re.M):
        cfg = m.group(1)
        r = paired(LORA[cfg], FROZ[cfg])
        got = (f"{LORA[cfg].mean():.4f}", f"{LORA[cfg].std(ddof=1):.4f}", f"{FROZ[cfg].mean():.4f}", f"{r['mean']:+.4f}",
               f"{r['sd_ddof1']:.4f}", str(r["n_pos"]), f"{r['p_t']:.4f}", f"{r['p_NB']:.4f}")
        n_j += check(f"internal check, LoRA - frozen {cfg}", got == m.groups()[1:], f"computed {got}")
    check("internal check: all 12 per-configuration lines parsed and reproduced", n_j == 12)
    for lab, pat, arr in (
            ("LoRA C1", r"LoRA C1 RetNet-ABMIL bestHP: diffs=\[[^\]]*\] mean=([+-][0-9.]+) sd1=([0-9.]+) pos=(\d)/5 "
                        r"p_t=([0-9.]+) p_NB=([0-9.]+)", C1L),
            ("frozen C1", r"frozen C1: diffs=\[[^\]]*\] mean=([+-][0-9.]+) sd1=([0-9.]+) pos=(\d)/5 p_t=([0-9.]+) p_NB=([0-9.]+)", C1F)):
        m = re.search(pat, JT)
        r = paired(arr)
        got = (f"{r['mean']:+.5f}", f"{r['sd_ddof1']:.4f}", str(r["n_pos"]), f"{r['p_t']:.4f}", f"{r['p_NB']:.4f}")
        check(f"internal check, {lab}", m is not None and got == m.groups(), f"computed {got}")
    m = re.search(r"difference-in-differences \(LoRA C1 - frozen C1\): mean=([+-][0-9.]+) sd1=([0-9.]+) p_t=([0-9.]+) "
                  r"p_NB=([0-9.]+)", JT)
    r = paired(C1L - C1F)
    got = (f"{r['mean']:+.5f}", f"{r['sd_ddof1']:.4f}", f"{r['p_t']:.4f}", f"{r['p_NB']:.4f}")
    check("internal check, difference-in-differences", m is not None and got == m.groups(), f"computed {got}")
else:
    print(f"  note: {J3CHK.name} (log of an earlier check; set CLAM_TR_LORA_CHECK_LOG) not found: "
          "this cross-check is skipped")

NFAIL = sum(not c["pass"] for c in CHECKS)
N_PRE = len(CHECKS)
print(f"\n== checks: {len(CHECKS) - NFAIL}/{len(CHECKS)} pass")
if NFAIL:
    print("STOP: a reproduction check failed; no new test is computed.")
    OUT_JSON.write_text(json.dumps({"status": "stopped", "checks": CHECKS}, indent=1), encoding="utf-8")
    sys.exit(1)

# ------------------------------------------------------------------ 7. planned tests
print("\n== 7. Planned Stage-4 tests (Appendix 1, app:lora_stats), computed on the existing folds")
A = {"C1_frozen": paired(FROZ["RetNet_bestHP"], FROZ["ABMIL_bestHP"]),
     "C1_lora": paired(LORA["RetNet_bestHP"], LORA["ABMIL_bestHP"]),
     "C1_change_lora_minus_frozen": paired(C1L - C1F)}
HALF = -0.5 * A["C1_frozen"]["mean"]  # change of the contrast that equals a reduction by exactly half
LO11 = {k: paired(LORA[f"{k[0]}_{k[2]}"], LORA[f"{k[1]}_{k[2]}"]) for k in KEYS}
assert LO11[KEYS[I_C1]]["diffs_fold0to4"] == A["C1_lora"]["diffs_fold0to4"]
LQ_T = bh([LO11[k]["p_t"] for k in KEYS])
LQ_NB = bh([LO11[k]["p_NB"] for k in KEYS])
check("BH: statsmodels fdr_bh == manual step-up (LoRA family, on p_t and on p_NB)",
      np.allclose(LQ_T, bh_manual([LO11[k]["p_t"] for k in KEYS]), atol=1e-15)
      and np.allclose(LQ_NB, bh_manual([LO11[k]["p_NB"] for k in KEYS]), atol=1e-15))
PC = {cfg: paired(LORA[cfg], FROZ[cfg]) for cfg in CFGS}
B6 = [f"{a}_bestHP" for a in AGGS]
C6_T, C6_NB = bh([PC[c]["p_t"] for c in B6]), bh([PC[c]["p_NB"] for c in B6])
C12_T, C12_NB = bh([PC[c]["p_t"] for c in CFGS]), bh([PC[c]["p_NB"] for c in CFGS])
ALT12_T = bh([LO11[k]["p_t"] for k in KEYS] + [A["C1_change_lora_minus_frozen"]["p_t"]])
ALT12_NB = bh([LO11[k]["p_NB"] for k in KEYS] + [A["C1_change_lora_minus_frozen"]["p_NB"]])
BONF = ALPHA / len(KEYS)
ALL = list(A.values()) + list(LO11.values()) + list(PC.values()) + list(FRO11.values())
MAX_DP_FS = max(abs(r["p_NB_foldsize_ratio"] - r["p_NB"]) for r in ALL)
MAX_DCI = max(max(abs(r["boot_ci95"][j] - r["boot_ci95_exact_enumeration"][j]) for j in (0, 1)) for r in ALL)


def line(lab, r, q_t=None, q_nb=None):
    qs = "" if q_t is None else f" q_t={q_t:.4f} q_NB={q_nb:.4f}"
    print(f"  {lab:46s} d={r['mean']:+.5f} sd={r['sd_ddof1']:.5f} pos={r['n_pos']}/5 CI=[{r['boot_ci95'][0]:+.4f},"
          f"{r['boot_ci95'][1]:+.4f}] p_t={r['p_t']:.4f} p_NB={r['p_NB']:.4f} p_W={r['p_W_exact']:.4f} "
          f"d_z={r['d_z']:+.3f}{qs}")


print(" A. Contrast 1 (RetNet_bestHP - ABMIL_bestHP, UNI); half-effect change = " + f"{HALF:+.5f}")
line("C1 on frozen UNI features", A["C1_frozen"], FQ_T[I_C1], FQ_NB[I_C1])
line("C1 on LoRA-UNI features", A["C1_lora"], LQ_T[I_C1], LQ_NB[I_C1])
line("C1 change, LoRA minus frozen (diff-in-diff)", A["C1_change_lora_minus_frozen"])
print(f"  C1 reduction 1 - LoRA/frozen = {RED:.4f}; Contrast 2 (UNI2-h): not computable, no LoRA-UNI2-h run")
print(" B. Eleven in-encoder contrasts on LoRA-UNI features (BH family m = 11), ascending p_t")
for i in sorted(range(len(KEYS)), key=lambda j: LO11[KEYS[j]]["p_t"]):
    a, b, h = KEYS[i]
    line(f"({ELEVEN[i][0]}) {a} vs {b} ({h})", LO11[KEYS[i]], LQ_T[i], LQ_NB[i])
print(" C. Per configuration, LoRA-UNI minus frozen UNI (q over the six bestHP; q12 over all twelve)")
for i, c in enumerate(CFGS):
    j6 = B6.index(c) if c in B6 else None
    qq = f" | q6_t={C6_T[j6]:.4f} q6_NB={C6_NB[j6]:.4f}" if j6 is not None else ""
    line(c, PC[c], C12_T[i], C12_NB[i])
    if qq:
        print(f"  {'':46s}{qq}")
print(f" Alternative family (eleven + diff-in-diff, m = 12): q for the diff-in-diff q_t={ALT12_T[-1]:.4f} "
      f"q_NB={ALT12_NB[-1]:.4f}; for (a) q_t={ALT12_T[I_C1]:.4f} q_NB={ALT12_NB[I_C1]:.4f}")
print(f" Sensitivity: NB with this split's fold sizes (factor 1/5 + {FOLD_RATIO:.5f}) changes no p_NB by more than "
      f"{MAX_DP_FS:.5f}; exact bootstrap enumeration (5^5 resamples) vs B = 10,000 differs by at most {MAX_DCI:.5f} in a CI bound")

# ------------------------------------------------------------------ 8. outputs
print("\n== 8. Outputs")
F = lambda x, nd=4: f"{x:+.{nd}f}"  # noqa: E731
P = lambda p: "$<0.001$" if p < 0.001 else f"{p:.4f}"  # noqa: E731


def tex_row(lab, r, q_t=None, q_nb=None):
    qt = "--" if q_t is None else f"{q_t:.4f}"
    qn = "--" if q_nb is None else f"{q_nb:.4f}"
    return (f"{lab} & ${F(r['mean'])}$ & $[{F(r['boot_ci95'][0])}, {F(r['boot_ci95'][1])}]$ & {P(r['p_t'])} & "
            f"{P(r['p_NB'])} & {P(r['p_W_exact'])} & ${F(r['d_z'], 3)}$ & {qt} & {qn} \\\\")


ORDER11 = sorted(range(len(KEYS)), key=lambda j: LO11[KEYS[j]]["p_t"])
ORDER6 = sorted(range(len(B6)), key=lambda j: PC[B6[j]]["p_t"])
LAB11 = [f"{a} vs {b} ({h})" for a, b, h in KEYS]
BONF_T = [LAB11[i] for i in ORDER11 if LO11[KEYS[i]]["p_t"] < BONF]
BONF_NB = [LAB11[i] for i in ORDER11 if LO11[KEYS[i]]["p_NB"] < BONF]


def listing(labels):
    if not labels:
        return "no row"
    return "only " + labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


CAP_BONF = (r". With the Bonferroni threshold $0.05/11 = 0.0045$ on panel~B, " + listing(BONF_T) + r" passes on $p_{t}$ and "
            + listing(BONF_NB) + r" on $p_{\mathrm{NB}}$")
TEX = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\footnotesize",
    r"\renewcommand{\arraystretch}{1.05}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption[Planned Stage~4 tests of the LoRA protocol, computed afterwards on the existing folds]{Planned Stage~4 tests "
    r"of the pre-registered LoRA protocol (\ref{app:lora_stats}), computed afterwards on the existing five folds; no model "
    r"was retrained. Frozen-UNI fold QWKs are recomputed from the per-fold prediction files of "
    r"Chapter~\ref{ch:bulgular} and LoRA-UNI fold QWKs are the Stage~4 values of \ref{app:lora_outcome}; every difference "
    r"pairs the two runs by fold index on identical fold slide sets (the two feature sets come from different patch "
    r"tilings, Chapter~\ref{ch:methodology} Section~\ref{sec:lora_v1}). $\Delta$~QWK: mean of the five fold-level "
    r"differences. 95\% CI: percentile "
    r"bootstrap over the five fold differences ($B=$~10,000 resamples, fixed seed). $p_{t}$: paired \textit{t}-test; "
    r"$p_{\mathrm{NB}}$: Nadeau--Bengio corrected test \citep{R073}, variance $(1/5+1/4)\,s^{2}$ with four degrees of "
    r"freedom; $p_{W}$: exact two-sided Wilcoxon signed-rank test (smallest attainable value 0.0625); $d_{z}$: mean "
    r"difference divided by the sample standard deviation of the differences. $q_{\mathrm{BH}}$ and "
    r"$q_{\mathrm{BH}}^{\mathrm{NB}}$: Benjamini--Hochberg \textit{q} on $p_{t}$ and on $p_{\mathrm{NB}}$; panel~B is "
    r"the planned family of eleven contrasts, the panel~A rows take their \textit{q} from Table~\ref{tab:decomp_stats_uni} "
    r"(frozen) and from panel~B (LoRA), the change row belongs to no family, and panel~C uses the family of its six rows, "
    r"which the plan does not specify; rows of panels~B and~C are sorted by ascending $p_{t}$" + CAP_BONF + "}",
    r"\label{tab:lora_planned_tests}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{l c c c c c c c c}",
    r"\hline",
    r"\textbf{Comparison} & \textbf{$\Delta$~QWK} & \textbf{95\% CI} & \textbf{$p_{t}$} & \textbf{$p_{\mathrm{NB}}$} & "
    r"\textbf{$p_{W}$} & \textbf{$d_{z}$} & \textbf{$q_{\mathrm{BH}}$} & \textbf{$q_{\mathrm{BH}}^{\mathrm{NB}}$} \\",
    r"\hline",
    r"\multicolumn{9}{l}{\textit{A. Pre-registered Contrast~1, \texttt{RetNet\_bestHP} minus \texttt{ABMIL\_bestHP} on UNI}} \\",
    r"\hline",
    tex_row("Frozen UNI features", A["C1_frozen"], FQ_T[I_C1], FQ_NB[I_C1]),
    tex_row("LoRA-UNI features", A["C1_lora"], LQ_T[I_C1], LQ_NB[I_C1]),
    tex_row("Change, LoRA minus frozen", A["C1_change_lora_minus_frozen"]),
    r"Contrast~2 (UNI2-h) & \multicolumn{8}{c}{not computable: no LoRA-UNI2-h run (Chapter~\ref{ch:oneriler} "
    r"Section~\ref{sec:f3})} \\",
    r"\hline",
    r"\multicolumn{9}{l}{\textit{B. Eleven in-encoder contrasts on LoRA-UNI features (Benjamini--Hochberg family, $m = 11$)}} \\",
    r"\hline",
] + [tex_row(f"{KEYS[i][0]} vs {KEYS[i][1]} ({KEYS[i][2]})", LO11[KEYS[i]], LQ_T[i], LQ_NB[i]) for i in ORDER11] + [
    r"\hline",
    r"\multicolumn{9}{l}{\textit{C. LoRA-UNI minus frozen UNI, per bestHP configuration}} \\",
    r"\hline",
] + [tex_row(f"{B6[j].split('_')[0]} (bestHP)", PC[B6[j]], C6_T[j], C6_NB[j]) for j in ORDER6] + [
    r"\hline",
    r"\end{tabular}}",
    r"\end{table}",
]
OUT_TEX.write_text("\n".join(TEX) + "\n", encoding="utf-8")
TEX_TEXT = OUT_TEX.read_text(encoding="utf-8")
check("LaTeX draft: no '{,}' separator, no em dash, no '$p$-' pattern, 9 columns per data row",
      "{,}" not in TEX_TEXT and "---" not in TEX_TEXT and "$p$-" not in TEX_TEXT
      and all(row.count(" & ") == 8 for row in TEX if row.endswith("\\\\") and "multicolumn" not in row))
print(f"  wrote {OUT_TEX.name}")

# ---- report (markdown); numbers formatted from the computed values, qualifiers derived from the data
c1l, c1f, dd = A["C1_lora"], A["C1_frozen"], A["C1_change_lora_minus_frozen"]
half_in = dd["boot_ci95"][0] <= HALF <= dd["boot_ci95"][1]
zero_in = dd["boot_ci95"][0] <= 0.0 <= dd["boot_ci95"][1]
nb_half_in = dd["ci95_NB"][0] <= HALF <= dd["ci95_NB"][1]
nb_zero_in = dd["ci95_NB"][0] <= 0.0 <= dd["ci95_NB"][1]
n11_nb = sum(q < ALPHA for q in LQ_NB)
n11_t = sum(q < ALPHA for q in LQ_T)
n11_bonf = sum(LO11[k]["p_t"] < BONF for k in KEYS)
i_min11 = min(range(len(KEYS)), key=lambda j: LQ_NB[j])
n6_nbp = sum(PC[c]["p_NB"] < ALPHA for c in B6)
n6_tp = sum(PC[c]["p_t"] < ALPHA for c in B6)
c6_minp = min(B6, key=lambda c: PC[c]["p_NB"])
n6_q = sum(q < ALPHA for q in C6_NB) + sum(q < ALPHA for q in C6_T)
assert c1l["n_pos"] == 5 and RED < 0.5 and c1l["mean"] > 0, "decision-rule template assumes sign kept and reduction < 1/2"


def words(n):
    return ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"][n]


def mdrow(lab, r, q_t=None, q_nb=None):
    qt = "--" if q_t is None else f"{q_t:.4f}"
    qn = "--" if q_nb is None else f"{q_nb:.4f}"
    return (f"| {lab} | {F(r['mean'])} | [{F(r['boot_ci95'][0])}, {F(r['boot_ci95'][1])}] | {r['p_t']:.4f} | "
            f"{r['p_NB']:.4f} | {r['p_W_exact']:.4f} | {F(r['d_z'], 3)} | {qt} | {qn} | {r['n_pos']}/5 |")


fold_rows = []
for a in [x for x in RANK_F]:
    c = f"{a}_bestHP"
    fold_rows.append(f"| {c} | frozen | " + " | ".join(f"{v:.4f}" for v in FROZ[c]) + f" | {FROZ[c].mean():.4f} |")
    fold_rows.append(f"|  | LoRA | " + " | ".join(f"{v:.4f}" for v in LORA[c]) + f" | {LORA[c].mean():.4f} |")
    fold_rows.append(f"|  | LoRA - frozen | " + " | ".join(F(v) for v in PC[c]["diffs_fold0to4"]) + f" | {F(PC[c]['mean'])} |")
fold_rows.append("| C1 = RetNet - ABMIL | frozen | " + " | ".join(F(v) for v in C1F) + f" | {F(C1F.mean())} |")
fold_rows.append("|  | LoRA | " + " | ".join(F(v) for v in C1L) + f" | {F(C1L.mean())} |")
fold_rows.append("|  | change | " + " | ".join(F(v) for v in C1L - C1F) + f" | {F((C1L - C1F).mean())} |")
n_checks = len(CHECKS)
S1 = (r"The pre-registered rule of \ref{app:lora_question} reads: ``" + RULE_QUOTE + r"'' On LoRA-UNI features "
      r"(Table~\ref{tab:lora_planned_tests}, panel~A) "
      rf"Contrast~1 is positive in all five folds (${F(c1l['mean'])}$, 95\% CI $[{F(c1l['boot_ci95'][0])}, "
      rf"{F(c1l['boot_ci95'][1])}]$; $p_{{t}} = {c1l['p_t']:.4f}$, $p_{{\mathrm{{NB}}}} = {c1l['p_NB']:.4f}$, "
      rf"$q_{{\mathrm{{BH}}}}^{{\mathrm{{NB}}}} = {LQ_NB[I_C1]:.4f}$), and its mean is {100 * RED:.1f}\% smaller than on frozen "
      rf"features (${F(c1f['mean'])}$), so the point estimates meet neither condition of the rule.")
S2T = (rf"The fold-paired change of the contrast, LoRA minus frozen, is ${F(dd['mean'])}$ (95\% CI "
       rf"$[{F(dd['boot_ci95'][0])}, {F(dd['boot_ci95'][1])}]$; $p_{{t}} = {dd['p_t']:.4f}$, "
       rf"$p_{{\mathrm{{NB}}}} = {dd['p_NB']:.4f}$, $p_{{W}} = {dd['p_W_exact']:.4f}$); a reduction by exactly half "
       rf"corresponds to a change of ${F(HALF)}$, which lies {'inside' if half_in else 'outside'} this interval"
       + (", while zero lies outside it" if (half_in and not zero_in) else
          (", as does zero" if (half_in and zero_in) else
           (", while zero lies inside it" if zero_in else ", as does zero")))
       + rf"; the corresponding Nadeau--Bengio interval $[{F(dd['ci95_NB'][0])}, {F(dd['ci95_NB'][1])}]$ "
       + ("contains both values." if (nb_half_in and nb_zero_in) else
          ("contains the half-effect value but not zero." if nb_half_in else
           ("contains zero but not the half-effect value." if nb_zero_in else "contains neither value."))))
IDX_T = [i for i in ORDER11 if LQ_T[i] < ALPHA]
if len(IDX_T) == 2 and KEYS[IDX_T[0]][:2] == KEYS[IDX_T[1]][:2]:
    NAMES_T = rf" (\texttt{{{KEYS[IDX_T[0]][0]}}} versus \texttt{{{KEYS[IDX_T[0]][1]}}} under both hyperparameter sets)"
else:
    NAMES_T = (" (" + "; ".join(LAB11[i] for i in IDX_T) + ")") if IDX_T else ""


def verb(n, sing, plur):
    return sing if n <= 1 else plur


S3T = (rf"Over the planned family of eleven LoRA-UNI contrasts, {words(n11_nb)} {verb(n11_nb, 'reaches', 'reach')} "
       rf"$q_{{\mathrm{{BH}}}}^{{\mathrm{{NB}}}} < 0.05$ (smallest {LQ_NB[i_min11]:.4f}), while with the naive test "
       rf"{words(n11_t)} {verb(n11_t, 'reaches', 'reach')} $q_{{\mathrm{{BH}}}} < 0.05$" + NAMES_T
       + rf"; {words(n6_nbp)} of the six per-configuration LoRA-minus-frozen differences {verb(n6_nbp, 'has', 'have')} "
       rf"$p_{{\mathrm{{NB}}}} < 0.05$ (smallest {PC[c6_minp]['p_NB']:.4f}, \texttt{{{c6_minp.split('_')[0]}}}), and the "
       rf"headline-QWK rule of \ref{{app:lora_decision}} is defined on the five-fold mean alone and is not changed by these "
       rf"tests.")
DEF_BIG = [c for c in CFGS if c.endswith("defaultHP") and abs(PC[c]["mean"]) > 0.001]
DEF_SMALL = [c for c in CFGS if c.endswith("defaultHP") and c not in DEF_BIG]
DEF_ROWS = ("; ".join(f"{c.split('_')[0]} {F(PC[c]['mean'])} (p_t {PC[c]['p_t']:.4f}, p_NB {PC[c]['p_NB']:.4f})"
                      for c in DEF_BIG)
            + f"; the other {words(len(DEF_SMALL))} within +/-{max(abs(PC[c]['mean']) for c in DEF_SMALL):.4f} "
              f"(p_NB >= {min(PC[c]['p_NB'] for c in DEF_SMALL):.4f})")
R2C_TEXT = ("and the bestHP values\n  equal the independent copy `r2c1_c2_results.json` (Drive files)" if HAVE_QWK_COPY else
            "and the independent copy\n  `r2c1_c2_results.json` was not available (cross-check skipped)")
LOG_TEXT = ("`lora_checks_out.txt`; reproduced line by line" if HAVE_CHECK_LOG else
            "`lora_checks_out.txt` was not available (cross-check skipped)")
CHECKED_THESIS = ("Table 9 (UNI bestHP and UNI2-h\nSpatialMultRetNet means/SDs); all eleven rows of Table 12 and its "
                  f"Nadeau-Bengio footnote\n({c1f['p_NB']:.4f} / {FQ_NB[I_C1]:.4f}); the Ch.3 bootstrap and "
                  "Nadeau-Bengio intervals (validates the bootstrap\nconvention); the Appendix 1.4 / Section 4.9 values "
                  "(0.9529 +/- 0.0026, +0.0087 to +0.0046, 48%, 0.9503/0.9501,\ntau 0.33, -0.0040/+0.0115, band, "
                  "Stage 2 0.9563 +/- 0.0030, TOST 0.031/0.081)" if SRC is not None else
                  "the checks against the printed thesis text\nwere skipped (thesis LaTeX sources not available)")
CHECKED_LOG = (f"; and the earlier check of the LoRA results\n(LoRA C1 {F(c1l['mean'], 5)}, 5/5, p_t {c1l['p_t']:.4f}, "
               f"p_NB {c1l['p_NB']:.4f})." if HAVE_CHECK_LOG else
               "; the earlier check of the LoRA results\nwas skipped (its log is not available).")
MD = f"""# Planned LoRA Stage-4 tests (thesis Appendix 1), computed on the existing folds

Written by `06-thesis-figures/analysis/lora_planned_tests.py` (`python -P`; read-only, no training) together with
`lora_planned_tests_results.json` and `tab_lora_planned_tests.tex`, next to the script (console log:
`lora_planned_tests_out.txt` when the output is redirected as in the script header).

## 1. Data, pairing, reproduction

- **LoRA-UNI Stage 4:** `05-experimental/lora_v1_results/v20_lora_features_results.json`
  (sha256 = README). No per-slide file exists; the list is in fold order because `run_stage_4()` builds it as
  `[fold_qwks_dict[str(f)] for f in range(5)]`, and the logged
  `Fold 0-4: QWK=` lines of its last run (SpatialAdd_bestHP) equal the stored list by index.
- **Folds:** `lora_v1_results/lora_fold{{0-4}}_predictions.json` (Stage-2 held-out sets; each reproduces its fold
  QWK in `lora_training_results.json`). Stage 4 used the same `load_labels()` split and logged "Available LoRA
  features: 9128" (folds {'/'.join(map(str, FOLD_N))}).
- **Frozen UNI:** QWK recomputed (sklearn) from
  `CLAM_TR_Results/v20_fullscale/UNI/predictions/{{cfg}}_fold{{k}}_predictions.json`. Each value
  reproduces a stored value (|diff| < 1e-9), each fold has the LoRA fold's slides and labels, {R2C_TEXT}. The stored lists are not in fold order
  (ABMIL_bestHP folds 0-4 sit at list positions {FROZ_POS['ABMIL_bestHP']}). Duplicate files, matching one used:
  {" | ".join(f"{k}, {'; '.join(v)}" for k, v in DUP_NOTES.items()) or "none"}.
- **Earlier check of the LoRA results:** {LOG_TEXT}.
- **Caveat (thesis Ch.2 sec:lora_v1):** same slides per fold, but the LoRA features come from a re-tiling (224 px,
  about 604 patches per slide; frozen: 256 px, median 457), and LoRA defaultHP used lr 1e-4 against 2e-4 (bestHP
  sets identical), so a LoRA-minus-frozen difference changes encoder state and patch set together.

Per-fold QWK, bestHP in frozen-UNI rank order (defaultHP in the out file and JSON):

| Configuration | Series | F0 | F1 | F2 | F3 | F4 | Mean |
|---|---|---|---|---|---|---|---|
""" + "\n".join(fold_rows) + f"""

**Checks: {N_PRE}/{N_PRE} pass before any new test is computed ({n_checks - NFAIL}/{n_checks} in total)**: {CHECKED_THESIS}{CHECKED_LOG}

## 2. Results

**Reading of the plan (most literal).** Headline = Contrast 1 compared between LoRA-adapted and frozen features,
i.e. the fold-paired change (LoRA C1 minus frozen C1); Contrast 2 needs a LoRA-UNI2-h run, which does not exist.
BH family = the eleven Ch.2 contrasts (a)-(k) on the LoRA-UNI features, m = 11, as in Table 12. Alternatives,
also computed: C1 on each feature set (rows 1-2 of A); per-configuration LoRA minus frozen (panel C, q over the
six); BH over the eleven plus the change, m = 12 (change q {ALT12_T[-1]:.4f} / {ALT12_NB[-1]:.4f}; row (a)
{ALT12_T[I_C1]:.4f} / {ALT12_NB[I_C1]:.4f}).
**Methods.** scipy `ttest_rel`, `wilcoxon(method="exact")` (two-sided); d_z = mean / SD(ddof=1); Nadeau-Bengio as
Ch.2 ((1/5 + 1/4) s^2, 4 df); bootstrap = folds resampled with replacement, B = 10,000, RandomState(42) per
comparison, percentile 2.5/97.5; BH = statsmodels `fdr_bh` (q_BH on p_t, q_BH^NB on p_NB).

Headline (panel A; panels B and C are in the LaTeX table of section 3, all rows with fold signs in the JSON):

| Contrast 1 (RetNet - ABMIL, bestHP, UNI) | Delta QWK | 95% CI | p_t | p_NB | p_W | d_z | q_BH | q_BH^NB | pos |
|---|---|---|---|---|---|---|---|---|---|
{mdrow("frozen UNI", c1f, FQ_T[I_C1], FQ_NB[I_C1])}
{mdrow("LoRA-UNI", c1l, LQ_T[I_C1], LQ_NB[I_C1])}
{mdrow("change, LoRA - frozen", dd)}

Panel B (LoRA family, m = 11): smallest q_BH^NB {LQ_NB[i_min11]:.4f} ({LAB11[i_min11]}); q_BH < 0.05 on p_t:
{"; ".join(f"{LAB11[i]} {LQ_T[i]:.4f}" for i in IDX_T) or "none"}. Panel C (six bestHP LoRA - frozen): smallest
p_NB {PC[c6_minp]['p_NB']:.4f} ({c6_minp}), smallest q over the six {min(C6_T):.4f} (p_t) / {min(C6_NB):.4f} (NB).
defaultHP, LoRA minus frozen (lr confound, not tabled): {DEF_ROWS}. Bonferroni 0.0045 on panel B:
{listing(BONF_T)} passes on p_t, {listing(BONF_NB)} on p_NB. Half-effect change = {F(HALF, 5)}
(-0.5 x frozen C1). Sensitivity: the fold-size Nadeau-Bengio factor (1/5 + {FOLD_RATIO:.5f}) moves no p_NB by
more than {MAX_DP_FS:.5f}; exact enumeration of the 5^5 bootstrap resamples moves no CI bound by more than
{MAX_DCI:.5f}.

## 3. LaTeX table draft (`tab_lora_planned_tests.tex`)

```latex
{TEX_TEXT.rstrip()}
```

## 4. Relative to the pre-registered decision rule (LaTeX-ready)

{S1}

{S2T}

{S3T}
"""
OUT_MD.write_text(MD, encoding="utf-8")
print(f"  wrote {OUT_MD.name}")

RESULTS = {
    "status": "ok", "generated": datetime.now().isoformat(timespec="seconds"),
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                 "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__},
    "methods": {"pairing": "fold index 0-4 on identical fold slide sets", "p_t": "scipy ttest_rel / ttest_1samp, two-sided",
                "p_W": "scipy wilcoxon(method='exact'), two-sided", "d_z": "mean difference / SD(differences, ddof=1)",
                "bootstrap": "folds resampled with replacement, B=10000, numpy RandomState(42) re-seeded per comparison, "
                             "np.percentile [2.5, 97.5]",
                "nadeau_bengio": "t = mean / (SD * sqrt(1/5 + 1/4)), df = 4 (thesis Ch.2 sec:stat_plan); sensitivity with "
                                 f"1/5 + mean(n_test/n_train) = 1/5 + {FOLD_RATIO:.6f}",
                "bh": "statsmodels multipletests(method='fdr_bh'), cross-checked against a manual step-up"},
    "inputs_sha256": INPUT_SHA, "frozen_files_used": FROZEN_FILES, "duplicate_prediction_files": DUP_NOTES,
    "fold_sizes": FOLD_N,
    "per_fold": {"lora_uni_stage4": {c: LORA[c].tolist() for c in CFGS}, "frozen_uni": {c: FROZ[c].tolist() for c in CFGS},
                 "frozen_uni_stored_list_position_of_fold": FROZ_POS, "frozen_uni2h_SpatialMultRetNet_bestHP": U2_SMR.tolist(),
                 "lora_stage2_direct": S2A.tolist()},
    "checks": CHECKS,
    "A_contrast1": {**A, "q_frozen_family_t_nb": [FQ_T[I_C1], FQ_NB[I_C1]], "q_lora_family_t_nb": [LQ_T[I_C1], LQ_NB[I_C1]],
                    "reduction_1_minus_lora_over_frozen": RED, "half_effect_change": HALF,
                    "half_effect_inside_boot_ci_of_change": half_in, "zero_inside_boot_ci_of_change": zero_in,
                    "contrast2_uni2h": "not computable: no LoRA-UNI2-h run"},
    "B_eleven_lora": [{"id": ELEVEN[i][0], "comparison": f"{KEYS[i][0]} vs {KEYS[i][1]} ({KEYS[i][2]})", **LO11[KEYS[i]],
                       "q_BH_t": LQ_T[i], "q_BH_NB": LQ_NB[i], "bonferroni_t_pass": LO11[KEYS[i]]["p_t"] < BONF}
                      for i in range(len(KEYS))],
    "B_eleven_frozen_reproduced": [{"id": ELEVEN[i][0], "comparison": f"{KEYS[i][0]} vs {KEYS[i][1]} ({KEYS[i][2]})",
                                    **FRO11[KEYS[i]], "q_BH_t": FQ_T[i], "q_BH_NB": FQ_NB[i]} for i in range(len(KEYS))],
    "C_lora_minus_frozen": [{"config": c, **PC[c], "q_BH_t_family12": C12_T[i], "q_BH_NB_family12": C12_NB[i],
                             "q_BH_t_family6_bestHP": C6_T[B6.index(c)] if c in B6 else None,
                             "q_BH_NB_family6_bestHP": C6_NB[B6.index(c)] if c in B6 else None,
                             "lr_confound": c.endswith("defaultHP")} for i, c in enumerate(CFGS)],
    "alt_family_eleven_plus_change": {"q_t_change": ALT12_T[-1], "q_NB_change": ALT12_NB[-1], "q_t_a": ALT12_T[I_C1],
                                      "q_NB_a": ALT12_NB[I_C1]},
    "sensitivity": {"max_abs_delta_p_NB_foldsizes": MAX_DP_FS, "max_abs_delta_boot_ci_exact_vs_mc": MAX_DCI},
    "kendall_tau_bestHP_ranks": TAU, "decision_sentences_latex": [S1, S2T, S3T],
}
OUT_JSON.write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")
print(f"  wrote {OUT_JSON.name}")
NFAIL = sum(not c["pass"] for c in CHECKS)
print(f"\n== final: {len(CHECKS) - NFAIL}/{len(CHECKS)} checks pass")
sys.exit(1 if NFAIL else 0)
