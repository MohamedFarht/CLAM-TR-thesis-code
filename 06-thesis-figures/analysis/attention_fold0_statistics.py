# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/attention_fold0_statistics.py
# Thesis output : fold-0 attention statistics of Section 3.6 (p.64) and the per-slide values plotted in Figure 15
# Inputs        : five per-slide files in analysis/data/ (fold-0 attention dumps and prediction files of UNI RetNet_bestHP and UNI2-h SpatialMultRetNet_bestHP, from the run's v20 results folder; not in the repository)
# Outputs       : attention_fold0_results.json and attention_fold0_per_slide.csv next to this script
# Run           : python -P 06-thesis-figures/analysis/attention_fold0_statistics.py
# Provenance    : adapted from the thesis analysis scripts; the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
r"""Fold-0 attention statistics of the interpretability analysis in thesis Chapter 3 (Section 3.6).

The cohort statistics, the failure-mode figure (Figure 15) and the t-SNE analysis are computed on fold 0 from the
per-slide attention dumps that load the SELECTED-epoch weights (best_model_state_dict) at the configuration's
attention temperature (0.7, best-HP), i.e. the same dumps as the three slides displayed in the thesis. An earlier
extraction (07-q1-sprint/extract_rank1_attention_colab.ipynb: last saved epoch, temperature 1.0) is read for
comparison only.

Inputs (copies in ./data of these files of the run's results folder CLAM_TR_Results/v20_fullscale/):
  UNI2-h/predictions_attention/SpatialMultRetNet_bestHP_fold0_attention.json
  UNI/predictions_attention/RetNet_bestHP_fold0_attention.json
  UNI2-h/predictions/SpatialMultRetNet_bestHP_fold0_predictions.json   (authority for pred/label)
  UNI/predictions/RetNet_bestHP_fold0_predictions.json                 (authority for pred/label)
  rank1_attention/rank1_attention_per_slide.csv                        (earlier extraction, comparison only)

Rank-1 weight of a slide = max of attn_softmax (softmax over all N patches, before the TopK=128 selection);
the same definition as the earlier extraction (rank1_mass = max(A_softmax), cell 9 of that notebook).

Writes attention_fold0_results.json (raw, unrounded values) and attention_fold0_per_slide.csv next to this script.
Run: python -P attention_fold0_statistics.py
"""
import datetime
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import mannwhitneyu
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
F_UNI2H = DATA / "UNI2h_SpatialMultRetNet_bestHP_fold0_attention.json"
F_UNI = DATA / "UNI_RetNet_bestHP_fold0_attention.json"
P_UNI2H = DATA / "UNI2h_SpatialMultRetNet_bestHP_fold0_predictions.json"
P_UNI = DATA / "UNI_RetNet_bestHP_fold0_predictions.json"
OLD_CSV = DATA / "old_rank1_attention_per_slide.csv"
OUT_JSON = HERE / "attention_fold0_results.json"
OUT_CSV = HERE / "attention_fold0_per_slide.csv"

N_SLIDES = 1803
N_CLASSES = 6
THRESH = 0.9969                      # rank-1 level documented for the displayed examples (thesis Chapter 3)
# displayed slides of Fig. top_patches_per_grade: slide-id prefix printed in the figure, rank-1 value in caption
DISPLAYED = {1: ("db8ff1904b4d", "0.9987"), 3: ("32411bf01125", "0.9998"), 5: ("48ae1bdf6944", "0.9969")}
EXPECTED_CLASS_COUNTS = [520, 495, 218, 185, 173, 212]
K_NN = 15
N_PERM = 2000
# the exact t-SNE call of the earlier prediction-space figure script (UMAP is not installed, and the
# thesis caption says t-SNE)
TSNE_KW = dict(n_components=2, perplexity=40, random_state=42, init="pca", learning_rate="auto")
SAMPLE_SEED = 42                     # as in that script: rng = np.random.default_rng(42)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dump(path, experiment):
    """Load an attention dump, assert the data gates, return compact per-slide arrays (record order)."""
    with open(path) as f:
        d = json.load(f)
    hdr = {k: v for k, v in d.items() if k != "records"}
    assert d["experiment"] == experiment, d["experiment"]
    assert d["hp_name"] == "best", d["hp_name"]
    assert d["attn_temp"] == 0.7, d["attn_temp"]
    assert d["ckpt_key"] == "best_model_state_dict", d["ckpt_key"]
    assert d["fold"] == 0, d["fold"]
    assert d["n_failed"] == 0, d["n_failed"]
    assert len(d["records"]) == d["n_slides"] == N_SLIDES, (len(d["records"]), d["n_slides"])
    ids, npat, pred, lab, r1, top5 = [], [], [], [], [], []
    eff, mass5, mass128 = [], [], []
    max_sum_dev = 0.0
    n_sum_dev_gt_1e3 = 0
    worst_sum = ("", 0.0, 0)
    dev_T1 = dev_T07 = 0.0
    top1_is_topk0 = 0
    for r in d["records"]:
        s = np.asarray(r["attn_softmax"], dtype=np.float64)
        assert s.size == r["n_patches"], (r["slide_id"], s.size, r["n_patches"])
        tot = float(s.sum())
        # The initial gate "sum within 1e-3" FAILS on 14 slides per dump (max 0.0018 / 0.00215): attn_softmax is
        # stored to 5 decimals, and slides with hundreds of near-zero weights lose up to n_patches * 5e-6 of mass.
        # Replacement gate (stronger, recorded in the output JSON): |sum - 1| within the worst-case
        # 5-decimal rounding bound AND attn_softmax equal to softmax(attn_raw) within 1e-5 on every patch.
        if abs(tot - 1.0) > 1e-3:
            n_sum_dev_gt_1e3 += 1
        if abs(tot - 1.0) > abs(worst_sum[1]):
            worst_sum = (r["slide_id"], tot - 1.0, int(r["n_patches"]))
        assert abs(tot - 1.0) <= s.size * 5e-6 + 1e-9, (r["slide_id"], tot)
        assert s.size >= 5, r["slide_id"]
        max_sum_dev = max(max_sum_dev, abs(tot - 1.0))
        srt = np.sort(s)[::-1]
        ids.append(r["slide_id"])
        npat.append(int(r["n_patches"]))
        pred.append(int(r["pred"]))
        lab.append(int(r["label"]))
        r1.append(float(srt[0]))
        top5.append(srt[:5].copy())
        eff.append(float(1.0 / np.sum(s ** 2)))       # effective number of patches (inverse Simpson)
        mass5.append(float(srt[:5].sum()))
        mass128.append(float(srt[:128].sum()))
        # attn_softmax must be the softmax of the stored (post-temperature) scores attn_raw
        a = np.asarray(r["attn_raw"], dtype=np.float64)
        e1 = np.exp(a - a.max())
        e1 /= e1.sum()
        e2 = np.exp((a - a.max()) / 0.7)
        e2 /= e2.sum()
        d1 = float(np.abs(e1 - s).max())
        assert d1 <= 1e-5, (r["slide_id"], d1)
        dev_T1 = max(dev_T1, d1)
        dev_T07 = max(dev_T07, float(np.abs(e2 - s).max()))
        tk = r["topk_indices"]
        if s[tk[0]] == srt[0]:
            top1_is_topk0 += 1
    assert len(set(ids)) == N_SLIDES
    gates = {
        "header": hdr,
        "gate_header_ok": True,
        "gate_len_attn_softmax_eq_n_patches": True,
        "INITIAL_GATE_sum_within_1e-3": "FAILED" if n_sum_dev_gt_1e3 else "passed",
        "initial_gate_n_slides_sum_dev_gt_1e-3": n_sum_dev_gt_1e3,
        "max_abs_softmax_sum_dev": max_sum_dev,
        "worst_sum_slide_id_dev_npatches": list(worst_sum),
        "REPLACEMENT_GATE_sum_within_npatches_x_5e-6_and_softmax_of_attn_raw_within_1e-5": "passed",
        "max_abs_dev_stored_vs_softmax(attn_raw)": dev_T1,
        "info_max_abs_dev_stored_vs_softmax(attn_raw/0.7)": dev_T07,
        "info_slides_where_topk_indices[0]_holds_the_rank1_weight": top1_is_topk0,
    }
    out = dict(ids=np.array(ids), n_patches=np.array(npat), pred=np.array(pred), label=np.array(lab),
               r1=np.array(r1), top5=np.vstack(top5), eff=np.array(eff), mass5=np.array(mass5),
               mass128=np.array(mass128))
    return out, gates


def spread_stats(d):
    """Info for the 'spreads across the TopK=128 budget' sentence: how many patches carry the attention."""
    return {"median_effective_n_patches_inverse_simpson": float(np.median(d["eff"])),
            "p25_effective_n": float(np.percentile(d["eff"], 25)),
            "p75_effective_n": float(np.percentile(d["eff"], 75)),
            "share_effective_n_lt_2": float(np.mean(d["eff"] < 2)),
            "median_top5_mass": float(np.median(d["mass5"])),
            "median_top128_mass": float(np.median(d["mass128"])),
            "median_mass_outside_top128": float(np.median(1 - d["mass128"])),
            "median_n_patches": float(np.median(d["n_patches"]))}


def load_preds(path):
    with open(path) as f:
        p = json.load(f)
    assert len(p["slide_ids"]) == len(p["predictions"]) == len(p["labels"]) == N_SLIDES
    assert len(set(p["slide_ids"])) == N_SLIDES
    return {s: (int(a), int(b)) for s, a, b in zip(p["slide_ids"], p["predictions"], p["labels"])}


def gate_agreement(dump, preds):
    assert set(dump["ids"]) == set(preds), "slide sets differ between dump and saved predictions"
    pred_auth = np.array([preds[s][0] for s in dump["ids"]])
    lab_auth = np.array([preds[s][1] for s in dump["ids"]])
    pa = float(np.mean(pred_auth == dump["pred"]))
    la = float(np.mean(lab_auth == dump["label"]))
    assert pa >= 0.99 and la >= 0.99, (pa, la)
    return pred_auth, lab_auth, {"pred_agreement": pa, "label_agreement": la,
                                 "n_pred_mismatch": int(np.sum(pred_auth != dump["pred"])),
                                 "n_label_mismatch": int(np.sum(lab_auth != dump["label"]))}


def dist_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size), "median": float(np.median(x)), "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "mean": float(x.mean()), "min": float(x.min()),
            "max": float(x.max()), "count_ge_0.9969": int(np.sum(x >= THRESH)),
            "share_ge_0.9969": float(np.mean(x >= THRESH)), "count_ge_0.99": int(np.sum(x >= 0.99)),
            "count_ge_0.999": int(np.sum(x >= 0.999)), "count_ge_0.9": int(np.sum(x >= 0.9)),
            "count_ge_0.5": int(np.sum(x >= 0.5)), "share_ge_0.5": float(np.mean(x >= 0.5))}


def failure_mode(r1, pred, lab):
    correct = pred == lab
    rc, rw = r1[correct], r1[~correct]
    g = mannwhitneyu(rc, rw, alternative="greater")
    le = mannwhitneyu(rc, rw, alternative="less")
    two = mannwhitneyu(rc, rw, alternative="two-sided")
    out = {"n_correct": int(rc.size), "n_wrong": int(rw.size),
           "median_correct": float(np.median(rc)), "median_wrong": float(np.median(rw)),
           "mean_correct": float(rc.mean()), "mean_wrong": float(rw.mean()),
           "mwu_U_correct_vs_wrong": float(g.statistic),
           "mwu_p_one_sided_correct_gt_wrong": float(g.pvalue),
           "mwu_p_one_sided_correct_lt_wrong": float(le.pvalue),
           "mwu_p_two_sided": float(two.pvalue),
           "cles_P(correct>wrong)": float(g.statistic / (rc.size * rw.size)),
           "median_wrong_gt_median_correct": bool(np.median(rw) > np.median(rc)),
           "per_grade": []}
    for k in range(N_CLASSES):
        c = r1[(lab == k) & correct]
        w = r1[(lab == k) & ~correct]
        row = {"grade": k, "n_correct": int(c.size), "n_wrong": int(w.size),
               "median_correct": float(np.median(c)) if c.size else None,
               "median_wrong": float(np.median(w)) if w.size else None}
        if c.size and w.size:
            row["median_wrong_gt_median_correct"] = bool(row["median_wrong"] > row["median_correct"])
            row["diff_wrong_minus_correct"] = row["median_wrong"] - row["median_correct"]
            row["ratio_wrong_over_correct"] = row["median_wrong"] / row["median_correct"]
            row["mwu_p_two_sided"] = float(mannwhitneyu(c, w, alternative="two-sided").pvalue)
        out["per_grade"].append(row)
    inv = [r["grade"] for r in out["per_grade"] if r.get("median_wrong_gt_median_correct")]
    rev = [r["grade"] for r in out["per_grade"] if r.get("median_wrong_gt_median_correct") is False]
    out["grades_wrong_gt_correct"] = inv
    out["grades_correct_ge_wrong"] = rev
    if inv:
        out["grade_largest_diff_wrong_minus_correct"] = max(
            inv, key=lambda k: out["per_grade"][k]["diff_wrong_minus_correct"])
        out["grade_largest_ratio_wrong_over_correct"] = max(
            inv, key=lambda k: out["per_grade"][k]["ratio_wrong_over_correct"])
    return out


def knn_idx(Z, k):
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def purity(nb, y, n_perm, seed):
    N = y.size
    same = (y[nb] == y[:, None]).mean(1)
    per_class = [float(same[y == c].mean()) for c in range(N_CLASSES)]
    comp = [[float((y[nb[y == a]] == b).mean()) for b in range(N_CLASSES)] for a in range(N_CLASSES)]
    rng = np.random.default_rng(seed)
    null_overall = np.empty(n_perm)
    null_class = np.empty((n_perm, N_CLASSES))
    for i in range(n_perm):
        yp = rng.permutation(y)
        s = (yp[nb] == yp[:, None]).mean(1)
        null_overall[i] = s.mean()
        for c in range(N_CLASSES):
            null_class[i, c] = s[yp == c].mean()
    overall = float(same.mean())
    counts = np.bincount(y, minlength=N_CLASSES)
    out = {"k": int(nb.shape[1]), "overall_mean_purity": overall,
           "overall_chance": float(np.sum(counts * (counts - 1)) / (N * (N - 1))),
           "overall_perm_p": float((1 + np.sum(null_overall >= overall)) / (1 + n_perm)),
           "overall_null_p2.5": float(np.percentile(null_overall, 2.5)),
           "overall_null_p97.5": float(np.percentile(null_overall, 97.5)),
           "per_class_purity": per_class,
           "per_class_chance": [float((counts[c] - 1) / (N - 1)) for c in range(N_CLASSES)],
           "per_class_null_p2.5": [float(np.percentile(null_class[:, c], 2.5)) for c in range(N_CLASSES)],
           "per_class_null_p97.5": [float(np.percentile(null_class[:, c], 97.5)) for c in range(N_CLASSES)],
           "per_class_perm_p": [float((1 + np.sum(null_class[:, c] >= per_class[c])) / (1 + n_perm))
                                for c in range(N_CLASSES)],
           "neighbour_composition_row_point_class_col_neighbour_class": comp,
           "G2_neighbours_in_G3": comp[2][3], "G3_neighbours_in_G2": comp[3][2],
           "G2_G3_mutual_mean": (comp[2][3] + comp[3][2]) / 2,
           # expected neighbour composition under random labels: n_b/(N-1) for another class b, (n_a-1)/(N-1) same
           "chance_matrix": [[float((counts[b] - (1 if a == b else 0)) / (N - 1)) for b in range(N_CLASSES)]
                             for a in range(N_CLASSES)],
           "chance_G2_neighbours_in_G3": float(counts[3] / (N - 1)),
           "chance_G3_neighbours_in_G2": float(counts[2] / (N - 1)),
           "max_purity_excluding_G0_G1_G5": float(max(per_class[c] for c in (2, 3, 4))),
           "min_purity_cancer_grades": float(min(per_class[1:])),
           "max_purity_cancer_grades": float(max(per_class[1:])),
           "rank_classes_by_purity_desc": [int(c) for c in np.argsort(per_class)[::-1]],
           "top3_classes_by_purity": sorted(int(c) for c in np.argsort(per_class)[::-1][:3])}
    # most frequent foreign neighbour class for each class
    out["most_frequent_foreign_class"] = [int(max((b for b in range(N_CLASSES) if b != a),
                                                  key=lambda b: comp[a][b])) for a in range(N_CLASSES)]
    return out


def tsne_block(X, y, seed_perm):
    emb = TSNE(**TSNE_KW).fit_transform(X)
    Xs = StandardScaler().fit_transform(X)
    res = {"purity_2d_tsne": purity(knn_idx(emb, K_NN), y, N_PERM, seed_perm),
           "purity_5d_standardised": purity(knn_idx(Xs, K_NN), y, N_PERM, seed_perm),
           "silhouette_2d_tsne_true_grade": float(silhouette_score(emb, y)),
           "silhouette_5d_standardised_true_grade": float(silhouette_score(Xs, y)),
           "embedding_first3": emb[:3].tolist()}
    return res, emb


def stratified_sample(lab, n_per, seed):
    """Sampling of the earlier prediction-space figure script: per class in label order,
    rng.choice(len(sub), n_per, replace=False) if len(sub) > n_per."""
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(N_CLASSES):
        idx = np.flatnonzero(lab == k)
        if idx.size > n_per:
            idx = idx[rng.choice(idx.size, n_per, replace=False)]
        parts.append(idx)
    return np.concatenate(parts)


def main():
    res = {"meta": {"script": "06-thesis-figures/analysis/attention_fold0_statistics.py",
                    "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__, "pandas": pd.__version__,
                    "inputs_sha256": {p.name: sha256(p) for p in (F_UNI2H, F_UNI, P_UNI2H, P_UNI, OLD_CSV)},
                    "definition_rank1": "max of attn_softmax over all N patches (before TopK=128)",
                    "percentile_method": "numpy.percentile linear (default)",
                    "mwu": "scipy.stats.mannwhitneyu defaults (method auto), as in the earlier failure-mode figure script",
                    "tsne_call": TSNE_KW, "knn_k": K_NN, "n_perm": N_PERM}}

    # ---------------- gates ----------------
    d2, g2 = load_dump(F_UNI2H, "SpatialMultRetNet_bestHP")
    d1, g1 = load_dump(F_UNI, "RetNet_bestHP")
    p2 = load_preds(P_UNI2H)
    p1 = load_preds(P_UNI)
    pred2, lab2, a2 = gate_agreement(d2, p2)
    pred1, lab1, a1 = gate_agreement(d1, p1)
    g2.update(a2)
    g1.update(a1)
    assert set(d1["ids"]) == set(d2["ids"]), "the two dumps cover different fold-0 slides"
    assert np.array_equal(np.array([p1[s][1] for s in d2["ids"]]), lab2), "labels differ between configurations"

    disp = {}
    for g, (prefix, shown) in DISPLAYED.items():
        m = np.flatnonzero((lab2 == g) & (pred2 == g))
        best = m[np.argmax(d2["r1"][m])]
        v = float(d2["r1"][best])
        ties = int(np.sum(d2["r1"][m] == v))
        assert d2["ids"][best].startswith(prefix), (g, d2["ids"][best], prefix)
        assert f"{v:.4f}" == shown, (g, v, shown)
        assert ties == 1, (g, ties)
        srt = np.sort(d2["top5"][best])[::-1]
        disp[g] = {"slide_id": str(d2["ids"][best]), "rank1_stored": v, "rank2_stored": float(srt[1]),
                   "n_correct_slides_of_grade": int(m.size), "ties_at_max": ties,
                   "rounds_to": shown,
                   "rank_from_top_all_1803": int(1 + np.sum(d2["r1"] > v)),
                   "share_of_1803_strictly_below": float(np.mean(d2["r1"] < v)),
                   "share_of_1803_at_or_below": float(np.mean(d2["r1"] <= v))}
    g2["gate_displayed_slides_reproduced"] = True
    res["gates"] = {"UNI2h_SpatialMultRetNet_bestHP": g2, "UNI_RetNet_bestHP": g1}
    res["displayed_slides_UNI2h"] = disp

    # ---------------- A: UNI2-h SpatialMultRetNet_bestHP cohort statistics (fold 0) ----------------
    r1_2 = d2["r1"]
    corr2 = pred2 == lab2
    A = {"all": dist_stats(r1_2), "correct_only": dist_stats(r1_2[corr2]),
         "wrong_only": dist_stats(r1_2[~corr2]),
         "min_displayed_rank1": min(v["rank1_stored"] for v in disp.values())}
    A["count_ge_min_displayed"] = int(np.sum(r1_2 >= A["min_displayed_rank1"]))
    A["share_ge_min_displayed"] = float(np.mean(r1_2 >= A["min_displayed_rank1"]))
    A["n_correct"] = int(corr2.sum())
    A["n_wrong"] = int((~corr2).sum())
    A["per_grade_correct_count_ge_0.9969"] = {int(k): int(np.sum((r1_2 >= THRESH) & corr2 & (lab2 == k)))
                                               for k in range(N_CLASSES)}
    A["per_grade_median_all"] = {int(k): float(np.median(r1_2[lab2 == k])) for k in range(N_CLASSES)}
    A["spread_info"] = spread_stats(d2)
    res["A_UNI2h_SpatialMultRetNet_bestHP_fold0"] = A

    # ---------------- B: UNI RetNet_bestHP failure mode (fold 0) ----------------
    res["B_UNI_RetNet_bestHP_fold0_failure_mode"] = failure_mode(d1["r1"], pred1, lab1)
    res["B_UNI_RetNet_bestHP_fold0_failure_mode"]["all"] = dist_stats(d1["r1"])
    res["B_UNI_RetNet_bestHP_fold0_failure_mode"]["spread_info"] = spread_stats(d1)

    # ---------------- C: t-SNE of the rank-1..rank-5 profile (UNI RetNet_bestHP, fold 0) ----------------
    counts = np.bincount(lab1, minlength=N_CLASSES)
    n_per = int(counts.min())
    sample = stratified_sample(lab1, n_per, SAMPLE_SEED)
    X = d1["top5"][sample]
    y = lab1[sample]
    C, emb = tsne_block(X, y, seed_perm=7)
    C.update({"fold0_class_counts": counts.tolist(),
              "class_counts_match_expected": counts.tolist() == EXPECTED_CLASS_COUNTS,
              "n_per": n_per, "n_sample": int(sample.size),
              "sample_slide_ids": [str(s) for s in d1["ids"][sample]],
              "sample_labels": y.tolist()})
    res["C_UNI_RetNet_bestHP_fold0_tsne"] = C

    # ---------------- D: old extraction (last epoch, T = 1.0), same fold-0 slides ----------------
    old = pd.read_csv(OLD_CSV)
    Dres = {"csv_columns": list(old.columns)}

    def old_block(encoder, variant, ids):
        o = old[(old["encoder"] == encoder) & (old["variant"] == variant) & (old["hp"] == "best")
                & (old["fold"] == 0)]
        assert len(o) == N_SLIDES and o["slide_id"].is_unique, (encoder, variant, len(o))
        o = o.set_index("slide_id").loc[list(ids)]
        return o

    o2 = old_block("UNI2-h", "SpatialMultRetNet", d2["ids"])
    o1 = old_block("UNI", "RetNet", d1["ids"])
    or1_2 = o2["rank1_mass"].to_numpy(dtype=np.float64)
    or1_1 = o1["rank1_mass"].to_numpy(dtype=np.float64)
    Dres["A_old_UNI2h_SpatialMultRetNet_fold0"] = {"all": dist_stats(or1_2), "correct_only": dist_stats(or1_2[corr2])}
    Dres["A_paired_new_minus_old"] = {"share_new_gt_old": float(np.mean(r1_2 > or1_2)),
                                      "n_new_le_old": int(np.sum(r1_2 <= or1_2)),
                                      "median_diff": float(np.median(r1_2 - or1_2))}
    Dres["B_old_UNI_RetNet_fold0_failure_mode"] = failure_mode(or1_1, pred1, lab1)
    Dres["B_paired_new_minus_old"] = {"share_new_gt_old": float(np.mean(d1["r1"] > or1_1)),
                                      "n_new_le_old": int(np.sum(d1["r1"] <= or1_1)),
                                      "median_diff": float(np.median(d1["r1"] - or1_1))}
    Xold = o1[["rank1_mass", "rank2_mass", "rank3_mass", "rank4_mass", "rank5_mass"]].to_numpy(
        dtype=np.float64)[sample]
    Cold, _ = tsne_block(Xold, y, seed_perm=7)
    Dres["C_old_profile_same_sample"] = Cold
    res["D_old_extraction_fold0"] = Dres

    # per-slide table for the figure scripts' cross-checks and for the record
    pd.DataFrame({"slide_id": d1["ids"], "label": lab1, "pred_UNI_RetNet": pred1,
                  "rank1_UNI_RetNet_new": d1["r1"], "rank1_UNI_RetNet_old": or1_1,
                  **{f"rank{j + 1}_UNI_RetNet_new": d1["top5"][:, j] for j in range(5)}}).merge(
        pd.DataFrame({"slide_id": d2["ids"], "pred_UNI2h_SMRN": pred2, "rank1_UNI2h_SMRN_new": r1_2,
                      "rank1_UNI2h_SMRN_old": or1_2}), on="slide_id").to_csv(OUT_CSV, index=False)

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT_JSON, "and", OUT_CSV)

    # console summary
    print("gates:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "header"}
                                for k, v in res["gates"].items()}, indent=1))
    print("displayed:", json.dumps(disp, indent=1))
    for key in ("all", "correct_only"):
        print("A", key, json.dumps(A[key]))
    print("A spread", json.dumps(A["spread_info"]))
    print("B spread", json.dumps(res["B_UNI_RetNet_bestHP_fold0_failure_mode"]["spread_info"]))
    for sp in ("purity_2d_tsne", "purity_5d_standardised"):
        print("C perm p per class", sp, C[sp]["per_class_perm_p"])
    print("A old", json.dumps(Dres["A_old_UNI2h_SpatialMultRetNet_fold0"]["all"]))
    print("A paired", Dres["A_paired_new_minus_old"])
    B = res["B_UNI_RetNet_bestHP_fold0_failure_mode"]
    print("B", {k: v for k, v in B.items() if k not in ("per_grade", "all")})
    for row in B["per_grade"]:
        print("  B grade", row)
    Bo = Dres["B_old_UNI_RetNet_fold0_failure_mode"]
    print("B old", {k: v for k, v in Bo.items() if k != "per_grade"})
    for row in Bo["per_grade"]:
        print("  B old grade", row)
    print("B paired", Dres["B_paired_new_minus_old"])
    for tag, blk in (("C new", C), ("C old", Cold)):
        for sp in ("purity_2d_tsne", "purity_5d_standardised"):
            p = blk[sp]
            print(tag, sp, "overall", round(p["overall_mean_purity"], 4), "chance", round(p["overall_chance"], 4),
                  "perm_p", p["overall_perm_p"], "per_class", [round(v, 4) for v in p["per_class_purity"]],
                  "null97.5", [round(v, 4) for v in p["per_class_null_p97.5"]],
                  "G2->G3", round(p["G2_neighbours_in_G3"], 4), "G3->G2", round(p["G3_neighbours_in_G2"], 4),
                  "top3", p["top3_classes_by_purity"], "foreign", p["most_frequent_foreign_class"])
        print(tag, "silhouette 2d", round(blk["silhouette_2d_tsne_true_grade"], 4), "5d",
              round(blk["silhouette_5d_standardised_true_grade"], 4))
    print("C class counts", C["fold0_class_counts"], "n_per", C["n_per"], "match", C["class_counts_match_expected"])


if __name__ == "__main__":
    main()
