# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/sd_ddof1_and_t_intervals.py
# Thesis output : across-fold sample SD (ddof = 1) of QWK and accuracy (Table 9) and the Student-t 95% CIs of Table 10
# Inputs        : CLAM_TR_V20_DIR/{UNI,UNI2-h}/v20_final_summary.json (in data/) and per-slide prediction files predictions/{agg}_{hp}_fold{k}_predictions*.json (not in the repository)
# Outputs       : printed to stdout
# Run           : python -P 06-thesis-figures/analysis/sd_ddof1_and_t_intervals.py
# Provenance    : adapted from the thesis analysis scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Recompute, with ddof=1, every 5-fold mean+/-SD reported in the thesis, plus the Table 10 Student-t CIs
(ddof=1), Cohen's d (both ddof) and Hedges g.
All values come from the raw per-fold prediction JSONs; no value is rescaled from a rounded number.
"""
import json, os
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.environ.get("CLAM_TR_V20_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v20_fullscale"))  # repo path
VARIANTS = ["ABMIL", "DualBranch", "RetNet", "SpatialAdd", "SpatialMult", "SpatialMultRetNet"]
T4 = 2.776  # t_{0.975, df=4}

def cell(enc, variant, hp):
    qwk, acc = [], []
    for f in range(5):
        p = os.path.join(BASE, enc, "predictions", f"{variant}_{hp}HP_fold{f}_predictions.json")
        j = json.load(open(p))
        t = np.array(j["labels"]); pr = np.array(j["predictions"])
        qwk.append(cohen_kappa_score(t, pr, weights="quadratic"))
        acc.append(accuracy_score(t, pr))
    return np.array(qwk), np.array(acc)

for enc in ["UNI", "UNI2-h"]:
    print(f"\n=== {enc} bestHP : QWK ddof0->ddof1, acc ddof0->ddof1, Table-10 t-CI(ddof1) ===")
    print(f"{'variant':<18} {'QWKmean':>8} {'sd0':>7} {'sd1':>7} | {'t-CI(ddof1)':>20} | {'accMean':>8} {'accsd0':>7} {'accsd1':>7}")
    for v in VARIANTS:
        try:
            q, a = cell(enc, v, "best")
        except Exception as e:
            print(f"{v:<18} ERR {e}"); continue
        hw = T4 * q.std(ddof=1) / np.sqrt(5)
        print(f"{v:<18} {q.mean():8.4f} {q.std(ddof=0):7.4f} {q.std(ddof=1):7.4f} | "
              f"[{q.mean()-hw:.4f},{q.mean()+hw:.4f}] | "
              f"{a.mean():8.4f} {a.std(ddof=0):7.4f} {a.std(ddof=1):7.4f}")

# Cohen's d / Hedges g : UNI RetNet vs ABMIL (the headline d=+1.295)
qr, _ = cell("UNI", "RetNet", "best")
qa, _ = cell("UNI", "ABMIL", "best")
diff = qr - qa
print("\n=== Cohen's d_z (paired, UNI RetNet-ABMIL) ===")
print(f"per-fold diff = {np.round(diff,4).tolist()}; mean={diff.mean():.4f}")
for dd in (0, 1):
    sd = diff.std(ddof=dd)
    print(f"  ddof={dd}: sd_diff={sd:.4f}  Cohen_dz={diff.mean()/sd:.4f}")
d1 = diff.mean() / diff.std(ddof=1)
J = 1 - 3/(4*4 - 1)            # Hedges small-sample correction, df=4
g = d1 * J
# approx 95% CI on d_z (paired): SE = sqrt(1/n + d^2/(2n)); n=5 ; t*SE
n = 5; se = np.sqrt(1/n + d1**2/(2*n)); lo, hi = d1 - T4*se, d1 + T4*se
print(f"  ddof=1 d_z={d1:.4f}; Hedges g = d_z*J(={J:.3f}) = {g:.4f}; approx 95% CI on d_z = [{lo:.3f}, {hi:.3f}]")
