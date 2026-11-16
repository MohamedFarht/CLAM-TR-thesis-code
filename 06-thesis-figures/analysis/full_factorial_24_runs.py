# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/full_factorial_24_runs.py
# Thesis output : QWK and accuracy mean, SD (ddof 0 and 1) of all 24 runs (Appendix Table 2)
# Inputs        : CLAM_TR_V20_DIR/{UNI,UNI2-h}/v20_final_summary.json (in data/) and per-slide prediction files predictions/{agg}_{hp}_fold{k}_predictions*.json (not in the repository)
# Outputs       : printed to stdout
# Run           : python -P 06-thesis-figures/analysis/full_factorial_24_runs.py
# Provenance    : adapted from the thesis analysis scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Full 24-cell ddof=1 recompute (best + default HP) for tab:full_factorial (Appendix Table 2)."""
import json, os
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
BASE = os.environ.get("CLAM_TR_V20_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v20_fullscale"))  # repo path
VARIANTS = ["ABMIL", "DualBranch", "RetNet", "SpatialAdd", "SpatialMult", "SpatialMultRetNet"]
def cell(enc, variant, hp):
    qwk, acc = [], []
    for f in range(5):
        p = os.path.join(BASE, enc, "predictions", f"{variant}_{hp}HP_fold{f}_predictions.json")
        j = json.load(open(p)); t = np.array(j["labels"]); pr = np.array(j["predictions"])
        qwk.append(cohen_kappa_score(t, pr, weights="quadratic")); acc.append(accuracy_score(t, pr))
    return np.array(qwk), np.array(acc)
print(f"{'enc':6} {'variant':18} {'hp':7} {'QWKmean':>8} {'qsd0':>6} {'qsd1':>6} {'accmean':>8} {'asd0':>6} {'asd1':>6}")
for enc in ["UNI", "UNI2-h"]:
    for hp in ["default", "best"]:
        for v in VARIANTS:
            try:
                q, a = cell(enc, v, hp)
            except Exception as e:
                print(f"{enc:6} {v:18} {hp:7} ERR {e}"); continue
            print(f"{enc:6} {v:18} {hp:7} {q.mean():8.4f} {q.std(ddof=0):6.4f} {q.std(ddof=1):6.4f} {a.mean():8.4f} {a.std(ddof=0):6.4f} {a.std(ddof=1):6.4f}")
