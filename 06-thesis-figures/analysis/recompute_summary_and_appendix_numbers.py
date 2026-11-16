# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/recompute_summary_and_appendix_numbers.py
# Thesis output : Appendix Table 2 (24 runs) and Appendix Table 3 (per-grade mean +/- SD); the numbers of the Summary and the Ozet
# Inputs        : CLAM_TR_V20_DIR/{UNI,UNI2-h}/v20_final_summary.json (in data/) and per-slide prediction files predictions/{agg}_{hp}_fold{k}_predictions*.json (not in the repository)
# Outputs       : printed to stdout
# Run           : python -P 06-thesis-figures/analysis/recompute_summary_and_appendix_numbers.py > recompute_summary_and_appendix_numbers_out.txt
# Provenance    : adapted from the thesis analysis scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Independent recomputation (read-only) of the v20 seed-42 factorial numbers used in the
abstracts and the appendix: per-cell QWK / accuracy / per-grade accuracy from the per-fold prediction
files, SD ddof=1, params, total time; 11-test families (naive paired t and Nadeau-Bengio), BH q
(statsmodels), Cohen's d_z, decay-only contrast, RM-ANOVA interaction (statsmodels AnovaRM).
Libraries: sklearn.metrics.cohen_kappa_score, scipy.stats, statsmodels.
Run: python -P recompute_summary_and_appendix_numbers.py > recompute_summary_and_appendix_numbers_out.txt
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.anova import AnovaRM

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.environ.get("CLAM_TR_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo path
BASE = os.environ.get("CLAM_TR_V20_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v20_fullscale"))  # repo path
ENCS = ["UNI", "UNI2-h"]
AGGS = ["ABMIL", "DualBranch", "RetNet", "SpatialAdd", "SpatialMult", "SpatialMultRetNet"]
HPS = ["defaultHP", "bestHP"]
K = 5
NBF = 1 / 5 + 1 / 4
LAB = list(range(6))

summ = {}
for e in ENCS:
    with open(os.path.join(BASE, e, "v20_final_summary.json"), encoding="utf-8") as fh:
        summ[e] = json.load(fh)

Q, A, G, NSL = {}, {}, {}, {}
notes = []
for e, a, h in itertools.product(ENCS, AGGS, HPS):
    ex = summ[e]["experiments"][f"{a}_{h}"]
    stored_q, stored_a = ex["fold_qwks"], ex["fold_accs"]
    qv, av, gv, nv = [], [], [], []
    for f in range(K):
        cands = sorted(glob.glob(os.path.join(BASE, e, "predictions", f"{a}_{h}_fold{f}_predictions*.json")))
        chosen = None
        for c in cands:
            with open(c, encoding="utf-8") as fh:
                d = json.load(fh)
            y = np.array(d["labels"], int)
            p = np.array(d["predictions"], int)
            q = cohen_kappa_score(y, p, weights="quadratic", labels=LAB)
            match = any(abs(q - s) < 1e-6 for s in stored_q)
            if len(cands) > 1:
                notes.append(f"{e} {a}_{h} fold{f}: candidate {os.path.basename(c)} qwk={q:.6f} match_stored={match}")
            if match and chosen is None:
                acc = float((y == p).mean())
                pg = [float((p[y == g] == g).mean()) if (y == g).any() else np.nan for g in LAB]
                chosen = (q, acc, pg, len(y))
        assert chosen is not None, (e, a, h, f)
        qv.append(chosen[0]); av.append(chosen[1]); gv.append(chosen[2]); nv.append(chosen[3])
    Q[(e, a, h)] = np.array(qv); A[(e, a, h)] = np.array(av); G[(e, a, h)] = np.array(gv); NSL[(e, a, h)] = nv
    # multiset checks against the stored summary
    assert sorted(np.round(qv, 9)) == sorted(np.round(stored_q, 9)), (e, a, h, "qwk")
    acc_ok = sorted(np.round(av, 9)) == sorted(np.round(stored_a, 9))
    if not acc_ok:
        notes.append(f"ACC MISMATCH {e} {a}_{h}: computed {np.round(sorted(av), 6)} stored {np.round(sorted(stored_a), 6)}")
    pgm = np.array([ex["per_grade_acc_mean"][str(g)] for g in LAB])
    if not np.allclose(np.nanmean(G[(e, a, h)], 0), pgm, atol=1e-9):
        notes.append(f"PER-GRADE MEAN MISMATCH {e} {a}_{h}: computed {np.round(np.nanmean(G[(e, a, h)], 0), 6)} stored {np.round(pgm, 6)}")

print("== notes")
for n in notes:
    print(n)

print("\n== Full factorial (mean +/- sd ddof=1; QWK 4dp, acc 4dp; params; total_time_min 1dp)")
for e in ENCS:
    for a in AGGS:
        for h in HPS:
            ex = summ[e]["experiments"][f"{a}_{h}"]
            q, ac = Q[(e, a, h)], A[(e, a, h)]
            print(f"{e:7s} {a:18s} {h:10s} QWK {q.mean():.4f} +/- {q.std(ddof=1):.4f} (sd0 {q.std():.4f}; raw mean {q.mean():.6f} sd1 {q.std(ddof=1):.6f}) | "
                  f"ACC {ac.mean():.4f} +/- {ac.std(ddof=1):.4f} (sd0 {ac.std():.4f}; raw {ac.mean():.6f} {ac.std(ddof=1):.6f}) | "
                  f"params {ex['params']:,} | time {ex['total_time_min']:.1f} (raw {ex['total_time_min']:.4f})")

print("\n== Per-grade accuracy bestHP (mean +/- sd ddof=1, 3dp)")
for e in ENCS:
    for a in AGGS:
        g = G[(e, a, "bestHP")]
        m, s = np.nanmean(g, 0), np.nanstd(g, 0, ddof=1)
        cells = " & ".join(f"{mm:.3f} $\\pm$ {ss:.3f}" for mm, ss in zip(m, s))
        print(f"{e:7s} {a:18s} {cells}")
        print(f"{'':7s} {'raw':18s} " + " ".join(f"{mm:.5f}+/-{ss:.5f}" for mm, ss in zip(m, s)))
        lo = int(np.argmin(m))
        print(f"{'':7s} {'min grade':18s} G{lo} ({m[lo]:.4f}); G0={m[0]:.4f} G5={m[5]:.4f}")

g0 = [np.nanmean(G[(e, a, 'bestHP')], 0)[0] for e in ENCS for a in AGGS]
g5 = [np.nanmean(G[(e, a, 'bestHP')], 0)[5] for e in ENCS for a in AGGS]
mins = [int(np.argmin(np.nanmean(G[(e, a, 'bestHP')], 0))) for e in ENCS for a in AGGS]
print(f"\nG0 range bestHP 12 rows: {min(g0):.4f} - {max(g0):.4f}; G5 range: {min(g5):.4f} - {max(g5):.4f}; "
      f"min-grade indices: {mins}; in G2/G3: {sum(m in (2, 3) for m in mins)}/12")
g0d = [np.nanmean(G[(e, a, 'defaultHP')], 0)[0] for e in ENCS for a in AGGS]
minsd = [int(np.argmin(np.nanmean(G[(e, a, 'defaultHP')], 0))) for e in ENCS for a in AGGS]
print(f"(defaultHP for reference: min-grade indices {minsd})")


def paired(d):
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    t_naive = stats.ttest_rel(d, np.zeros(n))  # same as one-sample on d
    p_naive = float(stats.ttest_1samp(d, 0.0).pvalue)
    tnb = m / (sd * np.sqrt(NBF))
    p_nb = float(2 * stats.t.sf(abs(tnb), n - 1))
    return m, sd, p_naive, p_nb, m / sd, int((d > 0).sum())


ROWS = [("RetNet", "ABMIL", "bestHP"), ("SpatialMult", "RetNet", "bestHP"), ("SpatialMult", "DualBranch", "bestHP"),
        ("SpatialMultRetNet", "SpatialMult", "bestHP"), ("RetNet", "ABMIL", "defaultHP"), ("SpatialAdd", "SpatialMult", "bestHP"),
        ("RetNet", "DualBranch", "bestHP"), ("DualBranch", "ABMIL", "bestHP"), ("SpatialMultRetNet", "RetNet", "bestHP"),
        ("SpatialAdd", "SpatialMult", "defaultHP"), ("SpatialMult", "DualBranch", "defaultHP")]
print("\n== 11-test families (fold-aligned), BH via statsmodels fdrcorrection")
for e in ENCS:
    res = []
    for a, b, h in ROWS:
        d = Q[(e, a, h)] - Q[(e, b, h)]
        res.append((f"{a} vs {b} ({h})",) + paired(d))
    qn = fdrcorrection([r[3] for r in res])[1]
    qnb = fdrcorrection([r[4] for r in res])[1]
    print(f"-- {e}: min q naive {qn.min():.4f}; min q NB {qnb.min():.4f}; n(q naive<0.05)={(qn < 0.05).sum()}; n(q NB<0.05)={(qnb < 0.05).sum()}; "
          f"Bonferroni passes naive: {sum(r[3] < 0.05 / 11 for r in res)}")
    for r, a1, a2 in sorted(zip(res, qn, qnb), key=lambda z: z[0][3]):
        print(f"   {r[0]:42s} D={r[1]:+.5f} sd1={r[2]:.5f} p_t={r[3]:.4f} p_NB={r[4]:.4f} dz={r[5]:+.3f} pos={r[6]}/5 q={a1:.4f} qNB={a2:.4f}")

print("\n== decay-only contrast SpatialMultRetNet - RetNet (bestHP)")
for e in ENCS:
    d = Q[(e, "SpatialMultRetNet", "bestHP")] - Q[(e, "RetNet", "bestHP")]
    m, sd, p, pn, dz, pos = paired(d)
    print(f"{e}: mean {m:+.5f} -> {m:+.4f}; p_t {p:.4f}; p_NB {pn:.4f}; pos {pos}/5")

print("\n== RM-ANOVA (statsmodels AnovaRM) bestHP, subject = fold, factors encoder x aggregator")
rows = []
for e in ENCS:
    for a in AGGS:
        for f in range(K):
            rows.append(dict(fold=f, enc=e, agg=a, qwk=Q[(e, a, "bestHP")][f]))
res = AnovaRM(pd.DataFrame(rows), "qwk", "fold", within=["enc", "agg"]).fit()
print(res.anova_table)

print("\n== HP gaps (best - default), 4dp means")
for e in ENCS:
    for a in AGGS:
        mb, md = Q[(e, a, "bestHP")].mean(), Q[(e, a, "defaultHP")].mean()
        print(f"{e:7s} {a:18s} {md:.4f} -> {mb:.4f} gap raw {mb - md:+.5f} (rounded-diff {round(mb, 4) - round(md, 4):+.4f})")

print("\n== ranges for appendix observation 2")
for e in ENCS:
    ret = [Q[(e, a, h)].mean() for a in ("RetNet", "SpatialMultRetNet") for h in HPS]
    sm = [Q[(e, "SpatialMult", h)].mean() for h in HPS]
    print(f"{e}: retention cells {np.round(sorted(ret), 4)}; SpatialMult cells {np.round(sorted(sm), 4)}")

print("\n== fold slide counts", {f: NSL[("UNI", "ABMIL", "bestHP")][f] for f in range(K)},
      "sum", sum(NSL[("UNI", "ABMIL", "bestHP")]))
