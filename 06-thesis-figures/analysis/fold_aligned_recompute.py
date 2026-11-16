# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/fold_aligned_recompute.py
# Thesis output : fold-aligned per-fold QWK of the 24 configurations; Tables 11-13 (contrasts, paired tests, BH-FDR, Nadeau-Bengio); repeated-measures ANOVA; Appendix Table 3 SDs
# Inputs        : CLAM_TR_V20_DIR/{UNI,UNI2-h}/v20_final_summary.json (in data/) and per-slide prediction files predictions/{agg}_{hp}_fold{k}_predictions*.json (not in the repository)
# Outputs       : printed to stdout
# Run           : python -P 06-thesis-figures/analysis/fold_aligned_recompute.py > fold_aligned_recompute_out.txt
# Provenance    : adapted from the thesis analysis scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Independent fold-aligned recomputation of the seed-42 v20 statistics quoted in the thesis.

Read-only. Source: the per-fold prediction files of the v20 runs
CLAM_TR_V20_DIR/{UNI,UNI2-h}/predictions/{agg}_{hp}_fold{k}_predictions*.json (results folder
CLAM_TR_Results/v20_fullscale) and v20_final_summary.json (stored fold_qwks, completion order).

Overrides (both described in 06-thesis-figures/TABLES_AND_FIGURES.md, section 2, points 2 and 3):
  * UNI2-h ABMIL_bestHP fold 0 -> summary value 0.949984 (prediction file was overwritten by the
    2026-05-02 retrain, 0.947576).  A second set of statistics with the retrained value is printed
    for comparison with the companion journal paper (which uses 0.9476).
  * UNI ABMIL_defaultHP fold 4 -> the file whose QWK matches the stored summary list.

Usage: python -P fold_aligned_recompute.py > fold_aligned_recompute_out.txt
"""
import glob
import itertools
import json
import os

import numpy as np
from scipy import stats

ROOT = os.environ.get("CLAM_TR_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo path
BASE = os.environ.get("CLAM_TR_V20_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v20_fullscale"))  # repo path
ENCS = ["UNI", "UNI2-h"]
AGGS = ["ABMIL", "DualBranch", "SpatialMult", "SpatialAdd", "RetNet", "SpatialMultRetNet"]
HPS = ["bestHP", "defaultHP"]
K = 5
NBF = 1 / 5 + 1 / 4
N = 6


def qwk(y, p, n=N):
    o = np.bincount(y * n + p, minlength=n * n).reshape(n, n).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    i, j = np.indices((n, n))
    w = (i - j) ** 2 / (n - 1) ** 2
    return 1 - (w * o).sum() / (w * e).sum()


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


summ = {e: load_json(os.path.join(BASE, e, "v20_final_summary.json")) for e in ENCS}

# ---------------------------------------------------------------- per-fold QWK from prediction files
Q = {}          # (enc, agg, hp) -> np.array of 5 fold QWKs in fold order
FILEQ = {}      # all candidate values incl. duplicates
slides = {}
for e, a, h in itertools.product(ENCS, AGGS, HPS):
    stored = summ[e]["experiments"][f"{a}_{h}"]["fold_qwks"]
    vals = []
    for f in range(K):
        cands = sorted(glob.glob(os.path.join(BASE, e, "predictions", f"{a}_{h}_fold{f}_predictions*.json")))
        cq = []
        for c in cands:
            d = load_json(c)
            y = np.array(d["labels"], int)
            p = np.array(d["predictions"], int)
            s = tuple(sorted(d["slide_ids"]))
            if (e, f) in slides:
                assert slides[(e, f)] == s, (e, a, h, f, c)
            else:
                slides[(e, f)] = s
            cq.append((os.path.basename(c), qwk(y, p)))
        FILEQ[(e, a, h, f)] = cq
        # choose the candidate that matches a stored value; else first
        chosen = None
        for name, v in cq:
            if any(abs(v - s) < 1e-6 for s in stored):
                chosen = v
                break
        if chosen is None:
            chosen = cq[0][1]
            print(f"WARNING no stored match {e} {a} {h} fold{f}: files {[(n, round(v, 6)) for n, v in cq]}")
        vals.append(chosen)
    Q[(e, a, h)] = np.array(vals)

# override UNI2-h ABMIL bestHP fold0 with the summary value (the value behind the thesis 0.9433)
stored_u2_abmil = summ["UNI2-h"]["experiments"]["ABMIL_bestHP"]["fold_qwks"]
unmatched = [s for s in stored_u2_abmil if not any(abs(s - v) < 1e-6 for v in Q[("UNI2-h", "ABMIL", "bestHP")])]
print("UNI2-h ABMIL bestHP stored values without a file match (local copy = original run?):", unmatched)
print("UNI2-h ABMIL bestHP fold0 local file QWK:", round(Q[("UNI2-h", "ABMIL", "bestHP")][0], 6))
Q_RETRAIN = dict(Q)
Q_RETRAIN[("UNI2-h", "ABMIL", "bestHP")] = Q[("UNI2-h", "ABMIL", "bestHP")].copy()
Q_RETRAIN[("UNI2-h", "ABMIL", "bestHP")][0] = 0.947576  # Drive file after the 2026-05-02 retrain

# slide sets: same per fold across encoders?
for f in range(K):
    print(f"fold {f}: n={len(slides[('UNI', f)])}; UNI==UNI2-h slide set: {slides[('UNI', f)] == slides[('UNI2-h', f)]}")

# ---------------------------------------------------------------- fold order of stored lists
print("\n== stored fold_qwks -> fold ids (completion order)")
n_inorder = 0
for e, a, h in itertools.product(ENCS, AGGS, HPS):
    stored = summ[e]["experiments"][f"{a}_{h}"]["fold_qwks"]
    ids = []
    for s in stored:
        m = [f for f in range(K) if abs(Q[(e, a, h)][f] - s) < 1e-6]
        ids.append(m[0] if len(m) == 1 else "?")
    inorder = ids == list(range(K))
    n_inorder += inorder
    print(f"{e:7s} {a + '_' + h:28s} order={ids} {'IN-ORDER' if inorder else ''}")
    # check means and std match
    st = summ[e]["experiments"][f"{a}_{h}"]
    assert abs(np.mean(stored) - st["mean_qwk"]) < 1e-9
print("stored lists in fold order:", n_inorder, "of 24")

# ---------------------------------------------------------------- per-config table
print("\n== per-config fold-ordered QWK (thesis convention: UNI2-h ABMIL fold0 = summary value)")
for e, a, h in itertools.product(ENCS, AGGS, HPS):
    v = Q[(e, a, h)]
    print(f"{e:7s} {a + '_' + h:28s} {np.round(v, 4)} mean={v.mean():.4f} sd1={v.std(ddof=1):.4f} sd0={v.std():.4f}")


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1)
    return out


def boot(diff, B=10000, seed=42):
    rng = np.random.RandomState(seed)
    b = []
    for _ in range(B):
        idx = rng.choice(len(diff), len(diff), replace=True)
        b.append(diff[idx].mean())
    return np.percentile(b, [2.5, 97.5])


def paired(d):
    m = d.mean()
    sd = d.std(ddof=1)
    n = len(d)
    t = m / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    tn = m / (sd * np.sqrt(NBF))
    pn = 2 * stats.t.sf(abs(tn), n - 1)
    pw = float(stats.wilcoxon(d, method="exact").pvalue)
    tc = stats.t.ppf(0.975, n - 1)
    ci_t = (m - tc * sd / np.sqrt(n), m + tc * sd / np.sqrt(n))
    ci_nb = (m - tc * sd * np.sqrt(NBF), m + tc * sd * np.sqrt(NBF))
    # exact sign-flip permutation
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    perm = np.abs((signs * d).mean(1)) >= abs(m) - 1e-15
    psf = perm.mean()
    return dict(m=m, sd=sd, p=p, pn=pn, pw=pw, dz=m / sd, pos=int((d > 0).sum()), ci_t=ci_t, ci_nb=ci_nb, psf=psf)


ROWS = [("RetNet", "ABMIL", "bestHP"), ("SpatialMult", "RetNet", "bestHP"), ("SpatialMult", "DualBranch", "bestHP"),
        ("SpatialMultRetNet", "SpatialMult", "bestHP"), ("RetNet", "ABMIL", "defaultHP"), ("SpatialAdd", "SpatialMult", "bestHP"),
        ("RetNet", "DualBranch", "bestHP"), ("DualBranch", "ABMIL", "bestHP"), ("SpatialMultRetNet", "RetNet", "bestHP"),
        ("SpatialAdd", "SpatialMult", "defaultHP"), ("SpatialMult", "DualBranch", "defaultHP")]


def family(QQ, e, label):
    out = []
    for a, b, h in ROWS:
        d = QQ[(e, a, h)] - QQ[(e, b, h)]
        r = paired(d)
        r["name"] = f"{a} vs {b} ({h})"
        r["d"] = d
        out.append(r)
    qn = bh([r["p"] for r in out])
    qnb = bh([r["pn"] for r in out])
    for r, a1, a2 in zip(out, qn, qnb):
        r["q"] = a1
        r["qnb"] = a2
    print(f"\n== 11-test family {e} ({label}); sorted by naive p")
    for r in sorted(out, key=lambda r: r["p"]):
        print(f"{r['name']:42s} D={r['m']:+.5f} p_t={r['p']:.4f} p_NB={r['pn']:.4f} p_W={r['pw']:.4f} p_sf={r['psf']:.4f} dz={r['dz']:+.3f} "
              f"pos={r['pos']}/5 q={r['q']:.4f} qNB={r['qnb']:.4f} sd1={r['sd']:.4f} "
              f"ciT=[{r['ci_t'][0]:+.4f},{r['ci_t'][1]:+.4f}] ciNB=[{r['ci_nb'][0]:+.4f},{r['ci_nb'][1]:+.4f}] Bonf(0.05/11=0.004545)={'PASS' if r['p'] < 0.05 / 11 else 'no'}")
    return out


FAM = {e: family(Q, e, "thesis values") for e in ENCS}
family(Q_RETRAIN, "UNI2-h", "retrained fold0 0.9476, paper convention")

# ---------------------------------------------------------------- factorial contrasts (tab:factorial_decomp)
print("\n== factorial contrasts (bestHP), mean +/- sd1 (sd0), v20 bootstrap RandomState(42) 10k percentile, t-CI, NB-CI")
for e in ENCS:
    DB = Q[(e, "DualBranch", "bestHP")]
    AB = Q[(e, "ABMIL", "bestHP")]
    RN = Q[(e, "RetNet", "bestHP")]
    SM = Q[(e, "SpatialMult", "bestHP")]
    SMR = Q[(e, "SpatialMultRetNet", "bestHP")]
    SA = Q[(e, "SpatialAdd", "bestHP")]
    con = {"Dual (DB-ABMIL)": DB - AB, "Enhancement (RetNet-DB)": RN - DB, "Spatial (SM-DB)": SM - DB,
           "Interaction (SMR-RN-SM+DB)": SMR - RN - SM + DB, "SMR-DB": SMR - DB, "SMR-ABMIL": SMR - AB,
           "RetNet-ABMIL": RN - AB, "SMR-SM": SMR - SM, "SpatialAdd-DB": SA - DB, "SpatialAdd-ABMIL": SA - AB,
           "SM-ABMIL": SM - AB}
    for k, d in con.items():
        lo, hi = boot(d)
        r = paired(d)
        print(f"{e:7s} {k:28s} mean={d.mean():+.5f} sd1={d.std(ddof=1):.4f} sd0={d.std():.4f} boot=[{lo:+.4f},{hi:+.4f}] "
              f"ciT=[{r['ci_t'][0]:+.4f},{r['ci_t'][1]:+.4f}] ciNB=[{r['ci_nb'][0]:+.4f},{r['ci_nb'][1]:+.4f}] dz={r['dz']:+.3f} p_t={r['p']:.4f} p_NB={r['pn']:.4f} pos={r['pos']}/5 diffs={np.round(d, 4)}")

# cross-encoder C2: RetNet(UNI) - ABMIL(UNI2-h)
for lab, QQ in (("thesis", Q), ("retrain", Q_RETRAIN)):
    d = QQ[("UNI", "RetNet", "bestHP")] - QQ[("UNI2-h", "ABMIL", "bestHP")]
    r = paired(d)
    print(f"\nC2 RetNet(UNI)-ABMIL(UNI2-h) [{lab}]: mean={r['m']:+.4f} pos={r['pos']}/5 p_t={r['p']:.4f} p_NB={r['pn']:.4f} diffs={np.round(d, 4)}")
d = Q[("UNI", "RetNet", "bestHP")] - Q[("UNI2-h", "SpatialMultRetNet", "bestHP")]
print(f"UNI RetNet - UNI2-h SMRN: mean={d.mean():+.4f} pos={(d > 0).sum()}/5")


# ---------------------------------------------------------------- two-way RM-ANOVA (fold = subject)
def rm_anova(QQ, hp="bestHP"):
    Y = np.zeros((2, 6, K))
    for i, e in enumerate(ENCS):
        for j, a in enumerate(AGGS):
            Y[i, j] = QQ[(e, a, hp)]
    gm = Y.mean()
    a_m = Y.mean((1, 2))
    b_m = Y.mean((0, 2))
    s_m = Y.mean((0, 1))
    ab_m = Y.mean(2)
    as_m = Y.mean(1)
    bs_m = Y.mean(0)
    nA, nB, nS = 2, 6, K
    SSA = nB * nS * ((a_m - gm) ** 2).sum()
    SSB = nA * nS * ((b_m - gm) ** 2).sum()
    SSS = nA * nB * ((s_m - gm) ** 2).sum()
    SSAB = nS * ((ab_m - a_m[:, None] - b_m[None, :] + gm) ** 2).sum()
    SSAS = nB * ((as_m - a_m[:, None] - s_m[None, :] + gm) ** 2).sum()
    SSBS = nA * ((bs_m - b_m[:, None] - s_m[None, :] + gm) ** 2).sum()
    SST = ((Y - gm) ** 2).sum()
    SSABS = SST - SSA - SSB - SSS - SSAB - SSAS - SSBS
    dfA, dfB, dfAB = 1, 5, 5
    dfAS, dfBS, dfABS = 4, 20, 20
    FA = (SSA / dfA) / (SSAS / dfAS)
    FB = (SSB / dfB) / (SSBS / dfBS)
    FAB = (SSAB / dfAB) / (SSABS / dfABS)
    return dict(FA=FA, pA=stats.f.sf(FA, dfA, dfAS), FB=FB, pB=stats.f.sf(FB, dfB, dfBS), FAB=FAB,
                pAB=stats.f.sf(FAB, dfAB, dfABS), etaAB=SSAB / (SSAB + SSABS), etaA=SSA / (SSA + SSAS))


for lab, QQ in (("thesis (fold0=0.9500)", Q), ("retrain (fold0=0.9476, paper)", Q_RETRAIN)):
    r = rm_anova(QQ)
    print(f"\nRM-ANOVA bestHP {lab}: encoder F(1,4)={r['FA']:.3f} p={r['pA']:.4f} eta={r['etaA']:.2f}; aggregator F(5,20)={r['FB']:.3f} p={r['pB']:.4f}; "
          f"interaction F(5,20)={r['FAB']:.3f} p={r['pAB']:.4f} partial eta2={r['etaAB']:.2f}")

# positional (submitted) version for reference
QP = {}
for e, a, h in itertools.product(ENCS, AGGS, HPS):
    QP[(e, a, h)] = np.array(summ[e]["experiments"][f"{a}_{h}"]["fold_qwks"])
r = rm_anova(QP)
print(f"RM-ANOVA bestHP POSITIONAL (stored lists): interaction F={r['FAB']:.3f} p={r['pAB']:.4f}; encoder F={r['FA']:.3f} p={r['pA']:.4f}")

# ---------------------------------------------------------------- leave-one-fold-out best aggregator
print("\n== leave-one-fold-out means (bestHP)")
for e in ENCS:
    for lo in range(K):
        keep = [f for f in range(K) if f != lo]
        ms = {a: Q[(e, a, "bestHP")][keep].mean() for a in AGGS}
        best = max(ms, key=ms.get)
        print(f"{e:7s} without fold {lo}: best={best} {ms[best]:.4f}; RetNet={ms['RetNet']:.4f} SMRN={ms['SpatialMultRetNet']:.4f}")

# ---------------------------------------------------------------- HP-set gaps (best - default means), timing, per-grade
print("\n== HP-set gaps (bestHP - defaultHP mean)")
gaps = []
for e in ENCS:
    for a in AGGS:
        b = summ[e]["experiments"][f"{a}_bestHP"]["mean_qwk"]
        dft = summ[e]["experiments"][f"{a}_defaultHP"]["mean_qwk"]
        gaps.append((b - dft, e, a, dft, b))
for g in sorted(gaps, reverse=True):
    print(f"{g[1]:7s} {g[2]:18s} gap={g[0]:+.4f} ({g[3]:.4f} -> {g[4]:.4f})")
print("\n== SpatialMult and full-retention ranges (means)")
for e in ENCS:
    sm = [summ[e]["experiments"][f"SpatialMult_{h}"]["mean_qwk"] for h in HPS]
    rn = [summ[e]["experiments"][f"{a}_{h}"]["mean_qwk"] for a in ("RetNet", "SpatialMultRetNet") for h in HPS]
    print(f"{e}: SpatialMult def/best {np.round(sm, 4)} range width {max(sm) - min(sm):.4f}; RetNet+SMRN 4 cells {np.round(rn, 4)} width {max(rn) - min(rn):.4f}")

print("\n== timing (total_time_min)")
for e in ENCS:
    tb = [summ[e]["experiments"][f"{a}_bestHP"]["total_time_min"] for a in AGGS]
    td = [summ[e]["experiments"][f"{a}_defaultHP"]["total_time_min"] for a in AGGS]
    print(f"{e}: bestHP sum={sum(tb):.1f} min ({sum(tb) / 60:.1f} h) median={np.median(tb):.1f}; defaultHP sum={sum(td):.1f}; all-12 median={np.median(tb + td):.1f} sum={sum(tb + td):.1f}")
    print("   bestHP times", {a: round(t, 1) for a, t in zip(AGGS, tb)})
    print("   default times", {a: round(t, 1) for a, t in zip(AGGS, td)})

print("\n== stds of best configs")
for e in ENCS:
    for a in AGGS:
        v = Q[(e, a, "bestHP")]
        print(f"{e:7s} {a:18s} mean={v.mean():.4f} sd1={v.std(ddof=1):.4f} sd0={v.std():.4f}")

print("\n== per-grade accuracy mean +/- sd1 (Appendix table)")
for e in ENCS:
    for a in AGGS:
        fm = summ[e]["experiments"][f"{a}_bestHP"]["fold_metrics"]
        pg = np.array([[fmm["per_grade_accuracy"][str(g)] for g in range(6)] for fmm in fm], float)
        print(f"{e:7s} {a:18s} " + " & ".join(f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(pg.mean(0), pg.std(0, ddof=1))))

print("\n== duplicate files")
for k, v in FILEQ.items():
    if len(v) > 1:
        print(k, [(n, round(q, 6)) for n, q in v])
