# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/verify_chapter3_numbers.py
# Thesis output : independent recomputation of the Chapter 3 numbers: Tables 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 and 19, the ANOVA F(5, 20) and the pooled out-of-fold values
# Inputs        : CLAM_TR_V20_DIR/{UNI,UNI2-h}/v20_final_summary.json (in data/) and per-slide prediction files predictions/{agg}_{hp}_fold{k}_predictions*.json (not in the repository); data/factorial_ci_inline.csv; CLAM_TR_PROJECT_DIR/PANDA/dataset/train.csv (PANDA label file, provider column; not in the repository)
# Outputs       : verify_ch3_out.json next to this script; log on stdout
# Run           : python -P 06-thesis-figures/analysis/verify_chapter3_numbers.py > verify_chapter3_numbers_out.txt
# Provenance    : adapted from the thesis analysis scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Independent recomputation of the Chapter 3 numbers (read-only).

Sources: v20_final_summary.json (UNI, UNI2-h), local per-fold prediction files, PANDA train.csv (provider),
sibling-repo bootstrap output (factorial_ci_inline.csv). Writes verify_ch3_out.json + prints a log.
"""
import glob
import itertools
import json
import os
import pathlib

import numpy as np
from scipy import stats

ROOT = os.environ.get("CLAM_TR_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo path
BASE = os.environ.get("CLAM_TR_V20_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v20_fullscale"))  # repo path
ENCS = ["UNI", "UNI2-h"]
AGGS = ["ABMIL", "DualBranch", "SpatialMult", "SpatialAdd", "RetNet", "SpatialMultRetNet"]
HPS = ["bestHP", "defaultHP"]
K = 5
NBF = 1 / 5 + 1 / 4
OUT = {}


def qwk_cm(o):
    n = o.shape[0]
    o = o.astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    i, j = np.indices((n, n))
    w = (i - j) ** 2 / (n - 1) ** 2
    return 1 - (w * o).sum() / (w * e).sum()


def cm6(y, p):
    return np.bincount(np.asarray(y) * 6 + np.asarray(p), minlength=36).reshape(6, 6)


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


summ = {e: load(os.path.join(BASE, e, "v20_final_summary.json")) for e in ENCS}

# ---------------- summary-derived per-config values
print("== summary-derived (mean/sd1 QWK, mean/sd1 acc, params, time, best epochs)")
S = {}
for e, a, h in itertools.product(ENCS, AGGS, HPS):
    x = summ[e]["experiments"][f"{a}_{h}"]
    fq = np.array(x["fold_qwks"])
    fa = np.array(x["fold_accs"])
    ep = np.array([m["best_epoch"] for m in x["fold_metrics"]], float)
    pg = np.array([[m["per_grade_accuracy"][str(g)] for g in range(6)] for m in x["fold_metrics"]], float)
    S[(e, a, h)] = dict(mean=float(fq.mean()), mean_stored=x["mean_qwk"], sd1=float(fq.std(ddof=1)),
                        sd0=float(fq.std()), acc=float(fa.mean()), acc_sd1=float(fa.std(ddof=1)),
                        params=x["params"], time=x["total_time_min"], ep=ep.tolist(),
                        ep_mean=float(ep.mean()), ep_sd1=float(ep.std(ddof=1)),
                        pg=pg.mean(0).tolist(), pg_stored=[x["per_grade_acc_mean"][str(g)] for g in range(6)])
    if h == "bestHP":
        s = S[(e, a, h)]
        print(f"{e:7s} {a:18s} qwk={s['mean']:.6f} sd1={s['sd1']:.6f} acc={s['acc']:.6f} acc_sd1={s['acc_sd1']:.6f} "
              f"params={s['params']} time={s['time']:.3f} ep={s['ep']} ep={s['ep_mean']:.2f}+-{s['ep_sd1']:.3f} "
              f"pg={[round(v, 4) for v in s['pg']]}")
OUT["summary"] = {f"{e}|{a}|{h}": v for (e, a, h), v in S.items()}

# ---------------- fold-aligned per-fold QWK from prediction files
Q, slides, ysl = {}, {}, {}
for e, a, h in itertools.product(ENCS, AGGS, HPS):
    stored = summ[e]["experiments"][f"{a}_{h}"]["fold_qwks"]
    vals = []
    for f in range(K):
        cands = sorted(glob.glob(os.path.join(BASE, e, "predictions", f"{a}_{h}_fold{f}_predictions*.json")))
        chosen = None
        for c in cands:
            d = load(c)
            v = qwk_cm(cm6(d["labels"], d["predictions"]))
            if any(abs(v - s) < 1e-6 for s in stored):
                chosen = (v, c, d)
                break
        assert chosen is not None, (e, a, h, f)
        vals.append(chosen[0])
        ids = tuple(chosen[2]["slide_ids"])
        if (e, f) in slides:
            assert set(slides[(e, f)]) == set(ids)
        else:
            slides[(e, f)] = ids
        ysl[(e, a, h, f)] = chosen[2]
    Q[(e, a, h)] = np.array(vals)
    assert sorted(np.round(vals, 9)) == sorted(np.round(stored, 9))
print("fold-aligned QWK reproduced for all 24 x 2 configs (multisets match stored lists)")

# ---------------- t CIs and bootstraps per config
T975 = stats.t.ppf(0.975, 4)
print(f"t_0.975,4 = {T975:.6f}; t/sqrt5 = {T975 / np.sqrt(5):.6f}; 2.776/sqrt5 = {2.776 / np.sqrt(5):.6f}")
sib = {}
import csv
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "factorial_ci_inline.csv"), encoding="utf-8") as fh:  # repo path
    for r in csv.DictReader(fh):
        sib[(r["encoder"], r["aggregator"])] = (float(r["ci_lo"]), float(r["ci_hi"]), int(r["n_folds"]))


def boot_default_rng(v, seed=42, B=10000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_rs(v, seed=42, B=10000):
    rng = np.random.RandomState(seed)
    b = [v[rng.choice(len(v), len(v), replace=True)].mean() for _ in range(B)]
    return np.percentile(b, [2.5, 97.5])


print("== per-config t-CI (exact t), half-width, sibling bootstrap CI, own bootstrap (default_rng) ")
CI = {}
for e in ENCS:
    for a in AGGS:
        v = Q[(e, a, "bestHP")]
        m, sd = v.mean(), v.std(ddof=1)
        hw = T975 * sd / np.sqrt(5)
        lo, hi = boot_default_rng(v)
        s = sib[(e, a)]
        CI[f"{e}|{a}"] = dict(lo=m - hw, hi=m + hw, hw=hw, boot=(lo, hi), sib=s)
        print(f"{e:7s} {a:18s} t=[{m - hw:.5f},{m + hw:.5f}] hw={hw:.5f} own_boot=[{lo:.5f},{hi:.5f}] "
              f"sib=[{s[0]:.5f},{s[1]:.5f}] n={s[2]} dlo={s[0] - (m - hw):+.4f} dhi={(m + hw) - s[1]:+.4f} "
              f"dwidth={(2 * hw) - (s[1] - s[0]):+.4f}")
OUT["ci"] = CI


# ---------------- paired family
def paired(d):
    m = d.mean()
    sd = d.std(ddof=1)
    n = len(d)
    t = m / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    pn = 2 * stats.t.sf(abs(m / (sd * np.sqrt(NBF))), n - 1)
    pw = float(stats.wilcoxon(d, method="exact").pvalue)
    tc = stats.t.ppf(0.975, n - 1)
    return dict(m=m, sd=sd, p=p, pn=pn, pw=pw, dz=m / sd, pos=int((d > 0).sum()),
                cint=(m - tc * sd / np.sqrt(n), m + tc * sd / np.sqrt(n)),
                cinb=(m - tc * sd * np.sqrt(NBF), m + tc * sd * np.sqrt(NBF)))


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1)
    return out


ROWS = [("RetNet", "ABMIL", "bestHP"), ("SpatialMult", "RetNet", "bestHP"), ("SpatialMult", "DualBranch", "bestHP"),
        ("SpatialMultRetNet", "SpatialMult", "bestHP"), ("RetNet", "ABMIL", "defaultHP"),
        ("SpatialAdd", "SpatialMult", "bestHP"), ("RetNet", "DualBranch", "bestHP"), ("DualBranch", "ABMIL", "bestHP"),
        ("SpatialMultRetNet", "RetNet", "bestHP"), ("SpatialAdd", "SpatialMult", "defaultHP"),
        ("SpatialMult", "DualBranch", "defaultHP")]
FAM = {}
for e in ENCS:
    rs = []
    for a, b, h in ROWS:
        r = paired(Q[(e, a, h)] - Q[(e, b, h)])
        r["name"] = f"{a} vs {b} ({h})"
        rs.append(r)
    qn, qnb = bh([r["p"] for r in rs]), bh([r["pn"] for r in rs])
    for r, x, y in zip(rs, qn, qnb):
        r["q"], r["qnb"] = x, y
    print(f"== family {e}")
    for r in sorted(rs, key=lambda r: r["p"]):
        print(f"  {r['name']:42s} D={r['m']:+.6f} p_t={r['p']:.5f} p_W={r['pw']:.4f} d={r['dz']:+.4f} q={r['q']:.5f} "
              f"pNB={r['pn']:.5f} qNB={r['qnb']:.5f} pos={r['pos']} sd={r['sd']:.5f} ciNB=[{r['cinb'][0]:+.5f},{r['cinb'][1]:+.5f}]")
    FAM[e] = {r["name"]: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rs}
OUT["family"] = FAM

# ---------------- factorial contrasts
print("== factorial contrasts (bestHP)")
FC = {}
for e in ENCS:
    g = lambda a: Q[(e, a, "bestHP")]
    con = {"Dual": g("DualBranch") - g("ABMIL"), "HeadsBlock": g("RetNet") - g("DualBranch"),
           "SpatialDecay": g("SpatialMult") - g("DualBranch"),
           "Interaction": g("SpatialMultRetNet") - g("RetNet") - g("SpatialMult") + g("DualBranch"),
           "Decay(SMR-RN)": g("SpatialMultRetNet") - g("RetNet")}
    for k, d in con.items():
        lo, hi = boot_rs(d)
        r = paired(d)
        FC[f"{e}|{k}"] = dict(mean=d.mean(), sd1=d.std(ddof=1), boot=[lo, hi], pos=r["pos"], diffs=d.tolist())
        print(f"  {e:7s} {k:14s} mean={d.mean():+.6f} sd1={d.std(ddof=1):.6f} boot=[{lo:+.6f},{hi:+.6f}] pos={r['pos']}/5")
OUT["contrasts"] = FC

# ---------------- ranks and differences
print("== ranks (bestHP mean QWK)")
for e in ENCS:
    order = sorted(AGGS, key=lambda a: -S[(e, a, "bestHP")]["mean"])
    print(f"  {e}: " + ", ".join(f"{a}={S[(e, a, 'bestHP')]['mean']:.6f}" for a in order))
    acc_order = sorted(AGGS, key=lambda a: -S[(e, a, "bestHP")]["acc"])
    print(f"  {e} by accuracy: " + ", ".join(f"{a}={S[(e, a, 'bestHP')]['acc']:.6f}" for a in acc_order))
LP = {"UNI": 0.8121, "UNI2-h": 0.8414}
for e in ENCS:
    for a in AGGS:
        print(f"  headroom {e} {a}: {S[(e, a, 'bestHP')]['mean'] - LP[e]:.6f}")
print(f"  RetNet(UNI)-ABMIL(UNI) = {S[('UNI', 'RetNet', 'bestHP')]['mean'] - S[('UNI', 'ABMIL', 'bestHP')]['mean']:.6f}")
print(f"  SMR(U2)-ABMIL(U2) = {S[('UNI2-h', 'SpatialMultRetNet', 'bestHP')]['mean'] - S[('UNI2-h', 'ABMIL', 'bestHP')]['mean']:.6f}")
print(f"  ABMIL UNI-U2 = {S[('UNI', 'ABMIL', 'bestHP')]['mean'] - S[('UNI2-h', 'ABMIL', 'bestHP')]['mean']:.6f}")
print(f"  RetNet UNI - SMR U2 = {S[('UNI', 'RetNet', 'bestHP')]['mean'] - S[('UNI2-h', 'SpatialMultRetNet', 'bestHP')]['mean']:.6f}")
u2ab = Q[("UNI2-h", "ABMIL", "bestHP")].copy()
u2ab[0] = 0.947576
print(f"  UNI2-h ABMIL retrain mean = {u2ab.mean():.6f}; local fold0 = {Q[('UNI2-h', 'ABMIL', 'bestHP')][0]:.6f}")
for e in ENCS:
    tb = [S[(e, a, "bestHP")]["time"] for a in AGGS]
    print(f"  time {e}: min={min(tb):.3f} max={max(tb):.3f} median={np.median(tb):.3f}")

# ---------------- pooled UNI RetNet bestHP
print("== pooled UNI RetNet_bestHP")
import csv as _csv
prov = {}
with open(os.path.join(ROOT, "PANDA", "dataset", "train.csv"), encoding="utf-8") as fh:
    for r in _csv.DictReader(fh):
        prov[r["image_id"]] = r["data_provider"]
Y, P, SID = [], [], []
for f in range(K):
    d = ysl[("UNI", "RetNet", "bestHP", f)]
    Y += d["labels"]
    P += d["predictions"]
    SID += d["slide_ids"]
Y, P = np.array(Y), np.array(P)
C = cm6(Y, P)
print(C)
n = C.sum()
po = np.trace(C) / n
print(f"n={n} correct={np.trace(C)} wrong={n - np.trace(C)} acc={po:.6f} pooledQWK={qwk_cm(C):.6f} "
      f"meanfold={Q[('UNI', 'RetNet', 'bestHP')].mean():.6f} diff={qwk_cm(C) - Q[('UNI', 'RetNet', 'bestHP')].mean():.6f}")
prec = np.diag(C) / C.sum(0)
rec = np.diag(C) / C.sum(1)
f1 = 2 * prec * rec / (prec + rec)
kap = []
for k in range(6):
    yb, pb = (Y == k).astype(int), (P == k).astype(int)
    o = np.bincount(yb * 2 + pb, minlength=4).reshape(2, 2).astype(float)
    p0 = np.trace(o) / o.sum()
    pe = (o.sum(1) * o.sum(0)).sum() / o.sum() ** 2
    kap.append((p0 - pe) / (1 - pe))
for k in range(6):
    print(f"  G{k} n={C[k].sum()} kappa={kap[k]:.5f} P={prec[k]:.5f} R={rec[k]:.5f} F1={f1[k]:.5f}")
print(f"  macro P={prec.mean():.5f} R={rec.mean():.5f} F1={f1.mean():.5f}")
OUT["pooled"] = dict(cm=C.tolist(), qwk=qwk_cm(C), acc=po, prec=prec.tolist(), rec=rec.tolist(), f1=f1.tolist(),
                     kappa=kap, macroP=prec.mean(), macroR=rec.mean(), macroF1=f1.mean())
pv = np.array([prov.get(s, "NA") for s in SID])
print("  provider counts:", {k: int((pv == k).sum()) for k in set(pv)})
for hosp in ["radboud", "karolinska"]:
    mk = pv == hosp
    c = cm6(Y[mk], P[mk])
    pr = np.diag(c) / np.where(c.sum(0) > 0, c.sum(0), 1)
    rc = np.diag(c) / np.where(c.sum(1) > 0, c.sum(1), 1)
    ff = np.where(pr + rc > 0, 2 * pr * rc / np.where(pr + rc > 0, pr + rc, 1), 0)
    print(f"  {hosp}: n={mk.sum()} QWK={qwk_cm(c):.6f} acc={np.trace(c) / c.sum():.6f} macroF1={ff.mean():.6f}")
    print("   ", c.tolist(), "colsum0?", (c.sum(0) == 0).tolist())
    OUT[f"prov_{hosp}"] = dict(n=int(mk.sum()), qwk=qwk_cm(c), acc=np.trace(c) / c.sum(), macroF1=ff.mean())

# ---------------- RM-ANOVA bestHP (fold = subject)
Y3 = np.zeros((2, 6, K))
for i, e in enumerate(ENCS):
    for j, a in enumerate(AGGS):
        Y3[i, j] = Q[(e, a, "bestHP")]


def rm(Y3):
    gm = Y3.mean()
    a_m, b_m, s_m = Y3.mean((1, 2)), Y3.mean((0, 2)), Y3.mean((0, 1))
    ab_m, as_m, bs_m = Y3.mean(2), Y3.mean(1), Y3.mean(0)
    SSA = 6 * K * ((a_m - gm) ** 2).sum()
    SSB = 2 * K * ((b_m - gm) ** 2).sum()
    SSS = 12 * ((s_m - gm) ** 2).sum()
    SSAB = K * ((ab_m - a_m[:, None] - b_m[None, :] + gm) ** 2).sum()
    SSAS = 6 * ((as_m - a_m[:, None] - s_m[None, :] + gm) ** 2).sum()
    SSBS = 2 * ((bs_m - b_m[:, None] - s_m[None, :] + gm) ** 2).sum()
    SSABS = ((Y3 - gm) ** 2).sum() - SSA - SSB - SSS - SSAB - SSAS - SSBS
    FA, FAB = (SSA / 1) / (SSAS / 4), (SSAB / 5) / (SSABS / 20)
    return FA, stats.f.sf(FA, 1, 4), FAB, stats.f.sf(FAB, 5, 20)


print("== RM-ANOVA thesis: FA=%.4f pA=%.4f FAB=%.4f pAB=%.4f" % rm(Y3))
Y3r = Y3.copy()
Y3r[1, 0, 0] = 0.947576
print("== RM-ANOVA retrain: FA=%.4f pA=%.4f FAB=%.4f pAB=%.4f" % rm(Y3r))

# ---------------- binomial
from math import comb
p1 = sum(comb(6, k) for k in (4, 5, 6)) / 64
print(f"binomial one-tailed {p1:.6f}; two-tailed {2 * p1:.6f}")

pathlib.Path(__file__).with_name("verify_ch3_out.json").write_text(json.dumps(OUT, indent=1, default=float),
                                                                    encoding="utf-8")
print("done")
