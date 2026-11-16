# ====================================================================================================
# CLAM-TR thesis code -- 06-thesis-figures/analysis/cohort_flow_counts.py
# Thesis output : PANDA cohort counts of Table 4, Table 5 (provider counts), Figure 3 and Figure 5 (the shipped output is data/cohort_flow_counts.json); its TCGA-PRAD part belongs to the companion journal paper
# Inputs        : under CLAM_TR_PROJECT_DIR: Kaggle-PANDA-1st-place-solution/.../input/*.csv and PANDA/dataset/train.csv (PANDA label files, not in the repository); TCGA-PRAD files under CLAM_TR_DRIVE_DIR; v20 prediction files under CLAM_TR_V20_DIR
# Outputs       : cohort_flow_counts.json and cohort_flow_counts_report.md next to this script
# Run           : python -I 06-thesis-figures/analysis/cohort_flow_counts.py
# Provenance    : adapted from the thesis analysis scripts (the same script is also released, for the companion
#                 journal paper, in the sibling repository CLAM-TR); the computation is unchanged. Lines marked
#                 "# repo path" read paths inside this repository or the environment variables described in
#                 06-thesis-figures/TABLES_AND_FIGURES.md.
# ====================================================================================================
"""Verify every number of the cohort flow diagram from the files themselves.

The PANDA column gives the counts of thesis Figure 3, Tables 4-5 and Figure 5; the TCGA-PRAD column belongs to the
cohort flow diagram of the companion journal paper.

PANDA (left column of the diagram)
  - release: train-5kfold.csv (and PANDA/dataset/train.csv if present) -> 10,616 slides
  - image-hash duplicate groups (duplicate_imgids_imghash_thres_090.csv): ids, groups, handling in s00_make_k_fold.py
    (duplicates are removed only for the stratification step and then co-assigned to the fold of the group's first
    image; they are NOT removed from the data set) -> checked empirically on train-5kfold.csv
  - model-based label denoising (s12_remove_noise_by_local_preds.py thresholds; README): 10,616 -> 9,128
  - five folds (kfold 1-5 -> 0-4): sizes, provider balance, ISUP distribution; cross-checked against the per-fold
    prediction files of the trained v20 runs (both encoders, all 6 aggregators, bestHP)
TCGA-PRAD (right column)
  - gdc_manifest.tsv (449 Diagnostic Slides) + clinical.csv -> 426 labelled slides (reproduce the 23 exclusions with
    the notebook's gleason_to_isup rule and list them); labels.csv equality check
  - labels_benign.csv (118 Solid Tissue Normal Tissue Slides); patients; sites; overlap with the DX cohort
  - feature folders and prediction files contain exactly these slides

Outputs: cohort_flow_counts.json, cohort_flow_counts_report.md.   Usage: python -I cohort_flow_counts.py
"""
import collections
import csv
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CLAM_TR_PROJECT_DIR", HERE.parents[1]))  # repo path
KAG = ROOT / 'Kaggle-PANDA-1st-place-solution' / 'Kaggle-PANDA-1st-place-solution-master'
F_KFOLD = KAG / 'input' / 'train-5kfold.csv'
F_DUP = KAG / 'input' / 'duplicate_imgids_imghash_thres_090.csv'
F_DENOISED = KAG / 'input' / 'train-5kfold_remove_noisy_by_0622_rad_13_08_ka_15_10.csv'
F_S00 = KAG / 'src' / 'data_process' / 's00_make_k_fold.py'
F_S12 = KAG / 'src' / 'data_process' / 's12_remove_noise_by_local_preds.py'
F_README = KAG / 'README.md'
F_TRAIN_ORIG = ROOT / 'PANDA' / 'dataset' / 'train.csv'
DRIVE = Path(os.environ.get('CLAM_TR_DRIVE_DIR', '/content/drive/MyDrive'))  # repo path
TCGA = DRIVE / 'TCGA_PRAD'  # repo path
F_MANIFEST = TCGA / 'gdc_manifest.tsv'
F_CLIN = TCGA / 'clinical.csv'
F_LABELS = TCGA / 'labels.csv'
F_BENIGN = TCGA / 'labels_benign.csv'
FEAT_DX = DRIVE / 'CLAM_TR_Features/TCGA_PRAD/uni_20x'  # repo path
FEAT_TS = DRIVE / 'CLAM_TR_Features/TCGA_PRAD/uni_20x_benign'  # repo path
PRED_DX = DRIVE / 'CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv'  # repo path
PRED_TS = DRIVE / 'CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv'  # repo path
V20 = Path(os.environ.get('CLAM_TR_V20_DIR', DRIVE / 'CLAM_TR_Results/v20_fullscale'))  # repo path
AGGS = ['ABMIL', 'DualBranch', 'SpatialMult', 'SpatialAdd', 'RetNet', 'SpatialMultRetNet']
OUT_JSON = HERE / 'cohort_flow_counts.json'  # repo path
OUT_MD = HERE / 'cohort_flow_counts_report.md'  # repo path


def rows(path, delim=','):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh, delimiter=delim))


def counter(it):
    return {str(k): v for k, v in sorted(collections.Counter(it).items(), key=lambda kv: str(kv[0]))}


def grep_lines(path, patterns):
    out = []
    for i, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if any(re.search(p, line) for p in patterns):
            out.append(f'L{i}: {line.strip()}')
    return out


def gleason_to_isup(p, s):
    """Exact copy of the rule in 07-q1-sprint/T2_1_phase2_tcga_prad_external_validation_colab.ipynb, cell 9."""
    if p is None or s is None:
        return None
    if (p, s) == (3, 3):
        return 1
    if (p, s) == (3, 4):
        return 2
    if (p, s) == (4, 3):
        return 3
    if (p, s) == (4, 4):
        return 4
    if (p, s) in [(3, 5), (5, 3)]:
        return 4
    if (p, s) in [(4, 5), (5, 4)]:
        return 5
    if (p, s) == (5, 5):
        return 5
    return None


def as_int(v):
    v = (v or '').strip()
    if v == '':
        return None
    return int(float(v))


def panda():
    out = {}
    kf = rows(F_KFOLD)
    ids = [r['image_id'] for r in kf]
    out['release'] = dict(file=str(F_KFOLD.relative_to(ROOT)), n_rows=len(kf), n_unique_ids=len(set(ids)),
                          per_provider=counter(r['data_provider'] for r in kf),
                          isup=counter(int(r['isup_grade']) for r in kf),
                          isup_by_provider={p: counter(int(r['isup_grade']) for r in kf if r['data_provider'] == p)
                                            for p in sorted({r['data_provider'] for r in kf})},
                          kfold_sizes=counter(int(r['kfold']) for r in kf))
    if F_TRAIN_ORIG.exists():
        tr = rows(F_TRAIN_ORIG)
        same = set(r['image_id'] for r in tr) == set(ids)
        lab_same = all(r['isup_grade'] == x['isup_grade'] for r, x in
                       zip(sorted(tr, key=lambda r: r['image_id']), sorted(kf, key=lambda r: r['image_id'])))
        kf_prov = {r['image_id']: r['data_provider'] for r in kf}
        out['release']['original_train_csv'] = dict(file=str(F_TRAIN_ORIG.relative_to(ROOT)), n_rows=len(tr),
                                                    same_id_set_as_kfold_csv=same, same_isup_labels=lab_same,
                                                    per_provider=counter(r['data_provider'] for r in tr),
                                                    provider_mismatches_vs_kfold_csv=sum(
                                                        1 for r in tr if kf_prov.get(r['image_id']) != r['data_provider']))
    fold_of = {r['image_id']: int(r['kfold']) for r in kf}
    # ---- duplicates ----
    dup = rows(F_DUP)
    dids = [r['image_id'] for r in dup]
    groups = collections.defaultdict(list)
    for r in dup:
        groups[r['group_id']].append(r)
    sizes = counter(len(v) for v in groups.values())
    excluded_from_strat = [r['image_id'] for r in dup if r['index_in_group'] != '0']
    in_release = sum(1 for i in set(dids) if i in fold_of)
    coassigned = sum(1 for g in groups.values() if len({fold_of[r['image_id']] for r in g if r['image_id'] in fold_of}) == 1)
    den = rows(F_DENOISED)
    den_ids = set(r['image_id'] for r in den)
    g_in_9128 = [sum(1 for r in g if r['image_id'] in den_ids) for g in groups.values()]
    out['duplicates'] = dict(
        file=str(F_DUP.relative_to(ROOT)), n_rows=len(dup), n_unique_ids=len(set(dids)), n_groups=len(groups),
        group_size_distribution=sizes, n_ids_not_first_in_group=len(excluded_from_strat),
        n_ids_present_in_release=in_release, n_groups_all_members_same_fold=coassigned,
        n_groups_split_across_folds=len(groups) - coassigned,
        n_dup_ids_in_9128=sum(1 for i in set(dids) if i in den_ids),
        n_groups_with_2plus_members_in_9128=sum(1 for x in g_in_9128 if x >= 2),
        n_slides_in_9128_belonging_to_multi_member_groups=sum(x for x in g_in_9128 if x >= 2),
        handling_in_code=grep_lines(F_S00, [r'img_ids_excluded', r'index_in_group', r'MultilabelStratifiedKFold\(',
                                            r'random_state=', r'"--seed"', r'Fill duplicated', r'gleason_score\.apply',
                                            r'data_provider\.apply']),
        reading='The duplicate list is used only for the fold assignment: non-first members of each image-hash '
                'group are left out of MultilabelStratifiedKFold (Gleason score x data provider, 5 splits, '
                'random_state 1222, shuffle) and then given the fold of the group\'s first image. No slide is '
                'removed at this step (all 10,616 remain in train-5kfold.csv).')
    # ---- denoising ----
    kf_by = {r['image_id']: r for r in kf}
    removed = [kf_by[i] for i in ids if i not in den_ids]
    s12 = Path(F_S12).read_text(encoding='utf-8')
    thr_rad = json.loads(re.search(r'thresholds_rad = (\[[^\]]*\])', s12).group(1))
    thr_ka = json.loads(re.search(r'thresholds_ka = (\[[^\]]*\])', s12).group(1))
    viol, maxgap = 0, collections.defaultdict(float)
    for r in den:
        isup = int(r['isup_grade'])
        thr = (thr_rad if r['data_provider'] == 'radboud' else thr_ka)[isup]
        gap = abs(isup - float(r['probs_raw']))
        maxgap[(r['data_provider'], isup)] = max(maxgap[(r['data_provider'], isup)], gap)
        if gap > thr + 1e-9:
            viol += 1
    same_fold = all(int(r['kfold']) == fold_of[r['image_id']] for r in den)
    same_isup = all(r['isup_grade'] == kf_by[r['image_id']]['isup_grade'] for r in den)
    prov_rel = collections.Counter(r['data_provider'] for r in kf)
    prov_rem = collections.Counter(r['data_provider'] for r in removed)
    out['denoising'] = dict(
        file=str(F_DENOISED.relative_to(ROOT)), n_kept=len(den), n_unique_kept=len(den_ids),
        kept_subset_of_release=den_ids <= set(ids), n_removed=len(removed),
        removed_per_provider=counter(r['data_provider'] for r in removed),
        removal_rate_per_provider={p: prov_rem[p] / prov_rel[p] for p in sorted(prov_rel)},
        removed_per_isup=counter(int(r['isup_grade']) for r in removed),
        removed_per_provider_isup={p: counter(int(r['isup_grade']) for r in removed if r['data_provider'] == p)
                                   for p in sorted(prov_rel)},
        removed_dup_ids=sum(1 for r in removed if r['image_id'] in set(dids)),
        thresholds_from_s12=dict(radboud_by_isup_0_to_5=thr_rad, karolinska_by_isup_0_to_5=thr_ka,
                                 rule='slide removed if |ISUP label - out-of-fold EfficientNet-b1 regression output| '
                                      '> threshold(provider, ISUP)'),
        file_name_encodes='rad_13_08 = Radboud 1.3 (ISUP 0 and 5) / 0.8 (ISUP 1-4); ka_15_10 = Karolinska 1.5 / 1.0',
        kept_rows_violating_threshold=viol,
        kept_max_gap_per_cell={f'{p}|{i}': round(v, 6) for (p, i), v in sorted(maxgap.items())},
        kept_kfold_equals_release_kfold=same_fold, kept_isup_equals_release_isup=same_isup,
        readme=grep_lines(F_README, [r'remove noise', r'Remove noise', r'did not fix seed', r'remove_noisy_by_0622',
                                     r'remove 20% Radboud', r'final_1\.yaml --kfold 1']))
    # ---- folds of the 9,128 ----
    folds = {}
    for k in range(1, 6):
        fr = [r for r in den if int(r['kfold']) == k]
        n_rad = sum(1 for r in fr if r['data_provider'] == 'radboud')
        folds[str(k - 1)] = dict(kfold_csv_value=k, n=len(fr), radboud=n_rad, karolinska=len(fr) - n_rad,
                                 pct_radboud=round(100 * n_rad / len(fr), 2),
                                 isup=counter(int(r['isup_grade']) for r in fr))
    out['cohort_9128'] = dict(n=len(den), per_provider=counter(r['data_provider'] for r in den),
                              isup=counter(int(r['isup_grade']) for r in den),
                              isup_by_provider={p: counter(int(r['isup_grade']) for r in den if r['data_provider'] == p)
                                                for p in sorted(prov_rel)},
                              folds=folds)
    # ---- cross-check against the trained runs ----
    xc = {}
    fold_sets = {k: {r['image_id'] for r in den if int(r['kfold']) == k + 1} for k in range(5)}
    for enc in ('UNI', 'UNI2-h'):
        pdir = V20 / enc / 'predictions'
        names = set(os.listdir(pdir))
        complete = [a for a in AGGS if all(f'{a}_bestHP_fold{k}_predictions.json' in names for k in range(5))]
        complete_def = [a for a in AGGS if all(f'{a}_defaultHP_fold{k}_predictions.json' in names for k in range(5))]
        per_fold = {}
        for k in range(5):
            with open(pdir / f'RetNet_bestHP_fold{k}_predictions.json', encoding='utf-8') as fh:
                d = json.load(fh)
            per_fold[str(k)] = dict(n=len(d['slide_ids']), equals_csv_fold=set(d['slide_ids']) == fold_sets[k])
        xc[enc] = dict(aggregators_with_5_bestHP_fold_files=complete, n_bestHP_configs=len(complete),
                       aggregators_with_5_defaultHP_fold_files=complete_def, retnet_bestHP_folds=per_fold)
    out['trained_runs_crosscheck'] = xc
    out['n_configurations_bestHP'] = sum(v['n_bestHP_configs'] for v in xc.values())
    return out


def tcga():
    out = {}
    man = rows(F_MANIFEST, '\t')
    clin = rows(F_CLIN)
    lab = rows(F_LABELS)
    ben = rows(F_BENIGN)
    sid = lambda fn: fn[:-4] if fn.lower().endswith('.svs') else fn
    man_ids = [sid(r['filename']) for r in man]
    slide_type = lambda s: re.match(r'TCGA-\w\w-\w{4}-(\d\d)(\w)-(\d\d)-(\w\w)(\w)', s)
    types = collections.Counter(slide_type(s).group(4) for s in man_ids)
    samples = collections.Counter(slide_type(s).group(1) + slide_type(s).group(2) for s in man_ids)
    out['gdc_manifest'] = dict(file=str(F_MANIFEST), n_rows=len(man), n_unique_slides=len(set(man_ids)),
                               slide_type_codes=dict(types), sample_codes=dict(samples),
                               n_patients=len({r['case_id'] for r in man}),
                               total_size_gb=round(sum(int(r['size']) for r in man) / 1e9, 2),
                               states=counter(r['state'] for r in man))
    cl = {r['submitter_id']: r for r in clin}
    out['clinical'] = dict(file=str(F_CLIN), n_cases=len(clin), n_unique_cases=len(cl),
                           n_cases_with_both_patterns=sum(1 for r in clin if as_int(r['primary_gleason_grade']) is not None
                                                          and as_int(r['secondary_gleason_grade']) is not None))
    included, excluded = [], []
    for r in man:
        s = sid(r['filename'])
        sub = r['case_id']
        c = cl.get(sub)
        if c is None:
            excluded.append(dict(slide_id=s, patient=sub, reason='case not in the GDC clinical export', primary=None, secondary=None))
            continue
        p, q = as_int(c['primary_gleason_grade']), as_int(c['secondary_gleason_grade'])
        isup = gleason_to_isup(p, q)
        if isup is None:
            reason = ('no primary/secondary Gleason pattern in the GDC clinical export' if (p is None or q is None)
                      else f'Gleason {p}+{q} outside the ISUP 2014 mapping (patterns 3-5 only)')
            excluded.append(dict(slide_id=s, patient=sub, reason=reason, primary=p, secondary=q,
                                 gleason_score_field=c['gleason_score'], grade_group_field=c['gleason_grade_group']))
        else:
            included.append(dict(slide_id=s, patient=sub, isup=isup))
    lab_ids = {r['slide_id']: int(r['isup_grade']) for r in lab}
    inc_ids = {r['slide_id']: r['isup'] for r in included}
    reasons = collections.Counter(e['reason'] for e in excluded)
    exc_pat = {e['patient'] for e in excluded}
    inc_pat = {r['patient'] for r in included}
    out['dx_exclusions'] = dict(
        rule='gleason_to_isup() of T2_1_phase2 notebook cell 9 applied to clinical.csv primary/secondary Gleason '
             'patterns; slide -> case via the manifest case_id (= submitter id)',
        n_manifest=len(man), n_included=len(included), n_excluded=len(excluded), reasons=dict(reasons),
        excluded_slides=excluded, n_excluded_patients=len(exc_pat),
        n_excluded_patients_with_other_included_slides=len(exc_pat & inc_pat),
        reproduces_labels_csv_ids=set(inc_ids) == set(lab_ids),
        reproduces_labels_csv_isup=all(inc_ids[k] == lab_ids.get(k) for k in inc_ids),
        labels_csv_rows=len(lab))
    pats = collections.Counter(r['patient'] for r in included)
    sites = collections.Counter(r['slide_id'].split('-')[1] for r in included)
    site_pat = collections.defaultdict(set)
    for r in included:
        site_pat[r['slide_id'].split('-')[1]].add(r['patient'])
    pat_isup = {}
    for r in included:
        pat_isup.setdefault(r['patient'], set()).add(r['isup'])
    out['dx_cohort'] = dict(
        n_slides=len(included), n_patients=len(pats), slides_per_patient=counter(pats.values()),
        isup_slide_level=counter(r['isup'] for r in included),
        isup_patient_level=counter(min(v) for v in pat_isup.values()),
        patients_with_conflicting_isup=sum(1 for v in pat_isup.values() if len(v) > 1),
        slide_type_codes=dict(collections.Counter(slide_type(r['slide_id']).group(4) for r in included)),
        dx_numbers=counter(slide_type(r['slide_id']).group(4) + slide_type(r['slide_id']).group(5) for r in included),
        n_sites=len(sites), slides_per_site=dict(sorted(sites.items(), key=lambda kv: -kv[1])),
        sites_with_n_slides_ge={str(t): sum(1 for v in sites.values() if v >= t) for t in (4, 8, 10, 20, 30)},
        slides_in_sites_with_n_ge={str(t): sum(v for v in sites.values() if v >= t) for t in (4, 8, 10, 20, 30)},
        sites_with_n_patients_ge={str(t): sum(1 for v in site_pat.values() if len(v) >= t) for t in (4, 8, 10, 20, 30)},
        sites_with_fewer_than_4_slides=sorted(k for k, v in sites.items() if v < 4))
    # ---- benign tissue slides ----
    bids = [r['slide_id'] for r in ben]
    bpat = collections.Counter(b[:12] for b in bids)
    bsites = collections.Counter(b.split('-')[1] for b in bids)
    btypes = collections.Counter(slide_type(b).group(4) for b in bids)
    bsamples = collections.Counter(slide_type(b).group(1) + slide_type(b).group(2) for b in bids)
    out['ts_benign'] = dict(
        file=str(F_BENIGN), n_slides=len(bids), n_unique_slides=len(set(bids)), n_patients=len(bpat),
        slide_type_codes=dict(btypes), sample_codes=dict(bsamples), isup_labels=counter(r['isup_grade'] for r in ben),
        n_sites=len(bsites), slides_per_site=dict(sorted(bsites.items(), key=lambda kv: -kv[1])),
        total_size_gb=round(sum(int(r['file_size_bytes']) for r in ben) / 1e9, 2),
        patients_also_in_dx426=len(set(bpat) & set(pats)), patients_also_in_manifest449=len(set(bpat) & {r['case_id'] for r in man}),
        query='GDC files: project TCGA-PRAD, cases.samples.sample_type = Solid Tissue Normal, data_format SVS '
              '(T2_2_phase3 notebook cell 9); all files returned were used')
    # ---- features and predictions present ----
    fdx = {f[:-3] for f in os.listdir(FEAT_DX) if f.endswith('.pt')}
    fts = {f[:-3] for f in os.listdir(FEAT_TS) if f.endswith('.pt')}
    pdx = {r['slide_id'] for r in rows(PRED_DX)}
    pts = {r['slide_id'] for r in rows(PRED_TS)}
    out['artefacts'] = dict(
        feature_dir_dx=str(FEAT_DX), n_feature_files_dx=len(fdx), dx_features_equal_labelled=fdx == set(lab_ids),
        feature_dir_ts=str(FEAT_TS), n_feature_files_ts=len(fts), ts_features_equal_labelled=fts == set(bids),
        n_predictions_dx=len(pdx), dx_predictions_equal_labelled=pdx == set(lab_ids),
        n_predictions_ts=len(pts), ts_predictions_equal_labelled=pts == set(bids))
    return out


def main():
    t0 = time.time()
    p = panda()
    t = tcga()
    flow = dict(
        panda_release=p['release']['n_rows'], panda_release_karolinska=p['release']['per_provider']['karolinska'],
        panda_release_radboud=p['release']['per_provider']['radboud'],
        dup_ids=p['duplicates']['n_unique_ids'], dup_groups=p['duplicates']['n_groups'],
        dup_nonfirst=p['duplicates']['n_ids_not_first_in_group'],
        dup_groups_coassigned=p['duplicates']['n_groups_all_members_same_fold'],
        denoise_removed=p['denoising']['n_removed'],
        denoise_removed_radboud=p['denoising']['removed_per_provider'].get('radboud', 0),
        denoise_removed_karolinska=p['denoising']['removed_per_provider'].get('karolinska', 0),
        denoise_thr_rad=p['denoising']['thresholds_from_s12']['radboud_by_isup_0_to_5'],
        denoise_thr_ka=p['denoising']['thresholds_from_s12']['karolinska_by_isup_0_to_5'],
        cohort=p['cohort_9128']['n'], cohort_karolinska=p['cohort_9128']['per_provider']['karolinska'],
        cohort_radboud=p['cohort_9128']['per_provider']['radboud'],
        cohort_isup=p['cohort_9128']['isup'],
        fold_sizes=[p['cohort_9128']['folds'][str(k)]['n'] for k in range(5)],
        fold_pct_radboud=[p['cohort_9128']['folds'][str(k)]['pct_radboud'] for k in range(5)],
        n_encoders=2, n_aggregators=len(AGGS), n_configurations=p['n_configurations_bestHP'],
        tcga_manifest=t['gdc_manifest']['n_rows'], tcga_manifest_patients=t['gdc_manifest']['n_patients'],
        tcga_excluded=t['dx_exclusions']['n_excluded'],
        tcga_excluded_no_gleason=sum(v for k, v in t['dx_exclusions']['reasons'].items() if k.startswith('no primary')),
        tcga_excluded_nonstandard=sum(v for k, v in t['dx_exclusions']['reasons'].items() if k.startswith('Gleason')),
        tcga_excluded_no_case=sum(v for k, v in t['dx_exclusions']['reasons'].items() if k.startswith('case not')),
        tcga_dx=t['dx_cohort']['n_slides'], tcga_dx_patients=t['dx_cohort']['n_patients'],
        tcga_dx_sites=t['dx_cohort']['n_sites'], tcga_dx_isup=t['dx_cohort']['isup_slide_level'],
        tcga_dx_sites_ge8=t['dx_cohort']['sites_with_n_slides_ge']['8'],
        tcga_dx_sites_ge20=t['dx_cohort']['sites_with_n_slides_ge']['20'],
        tcga_ts=t['ts_benign']['n_slides'], tcga_ts_patients=t['ts_benign']['n_patients'],
        tcga_ts_sites=t['ts_benign']['n_sites'], tcga_ts_patients_in_dx=t['ts_benign']['patients_also_in_dx426'])
    res = dict(meta=dict(script='06-thesis-figures/analysis/cohort_flow_counts.py', run=time.strftime('%Y-%m-%dT%H:%M:%S'),
                         runtime_s=round(time.time() - t0, 1)),
               panda=p, tcga=t, flow=flow)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding='utf-8')
    write_report(res)
    print(json.dumps(flow, indent=1))


def write_report(r):
    p, t, f = r['panda'], r['tcga'], r['flow']
    L = ['# Cohort flow counts verified from the files\n',
         f"Generated by `{r['meta']['script']}` on {r['meta']['run']}; all values in `cohort_flow_counts.json`. "
         "The `flow` block of the JSON holds the counts of the cohort flow diagram (PANDA: thesis Figure 3).\n",
         '## PANDA\n',
         '| Step | Count | Source / check |', '|---|---|---|',
         f"| Public release | {f['panda_release']} slides (Karolinska {f['panda_release_karolinska']}, Radboud {f['panda_release_radboud']}) | `train-5kfold.csv` ({p['release']['n_unique_ids']} unique ids)"
         + (f"; identical id set / ISUP labels / providers to `PANDA/dataset/train.csv`: {p['release']['original_train_csv']['same_id_set_as_kfold_csv']} / {p['release']['original_train_csv']['same_isup_labels']} / {p['release']['original_train_csv']['provider_mismatches_vs_kfold_csv'] == 0}" if 'original_train_csv' in p['release'] else '') + ' |',
         f"| Image-hash duplicate groups | {f['dup_ids']} ids in {f['dup_groups']} groups (sizes {p['duplicates']['group_size_distribution']}); {f['dup_nonfirst']} non-first members | `duplicate_imgids_imghash_thres_090.csv`; all ids in the release: {p['duplicates']['n_ids_present_in_release']} |",
         f"| Duplicate handling | not removed; {f['dup_groups_coassigned']} of {f['dup_groups']} groups have all members in one fold ({p['duplicates']['n_groups_split_across_folds']} split) | `s00_make_k_fold.py` (lines below) + empirical check on `train-5kfold.csv` |",
         f"| Model-based label denoising | -{f['denoise_removed']} slides (Radboud -{f['denoise_removed_radboud']} = {100 * p['denoising']['removal_rate_per_provider']['radboud']:.1f} %, Karolinska -{f['denoise_removed_karolinska']} = {100 * p['denoising']['removal_rate_per_provider']['karolinska']:.1f} %) | thresholds by ISUP 0-5: Radboud {f['denoise_thr_rad']}, Karolinska {f['denoise_thr_ka']} (`s12_remove_noise_by_local_preds.py`); kept rows violating the rule: {p['denoising']['kept_rows_violating_threshold']} |",
         f"| Training cohort | {f['cohort']} slides (Karolinska {f['cohort_karolinska']}, Radboud {f['cohort_radboud']}) | `train-5kfold_remove_noisy_by_0622_rad_13_08_ka_15_10.csv`; fold and ISUP identical to the release rows: {p['denoising']['kept_kfold_equals_release_kfold']} / {p['denoising']['kept_isup_equals_release_isup']} |",
         f"| Five folds (0-4) | {' / '.join(str(x) for x in f['fold_sizes'])} | % Radboud per fold {f['fold_pct_radboud']} |",
         f"| Configurations | {f['n_configurations']} ({f['n_encoders']} encoders x {f['n_aggregators']} aggregators, bestHP) | complete five-fold prediction files on Drive; RetNet fold sets equal the CSV folds: "
         + ', '.join(f"{enc} {all(v['equals_csv_fold'] for v in x['retnet_bestHP_folds'].values())}" for enc, x in p['trained_runs_crosscheck'].items()) + ' |',
         '\nRemoved by denoising, per ISUP: ' + json.dumps(p['denoising']['removed_per_isup']) + '; per provider x ISUP: ' + json.dumps(p['denoising']['removed_per_provider_isup']) + '.',
         '\nISUP distribution, release (10,616): ' + json.dumps(p['release']['isup']) + '; training cohort (9,128): ' + json.dumps(p['cohort_9128']['isup']) + '.',
         '\nDuplicate ids remaining in the 9,128: %d; groups with >= 2 members in the 9,128: %d (%d slides), co-assigned by construction.'
         % (p['duplicates']['n_dup_ids_in_9128'], p['duplicates']['n_groups_with_2plus_members_in_9128'],
            p['duplicates']['n_slides_in_9128_belonging_to_multi_member_groups']),
         '\n`s00_make_k_fold.py` lines that define the handling:\n']
    L += ['    ' + x for x in p['duplicates']['handling_in_code']]
    L += ['\nREADME lines on the denoising:\n'] + ['    ' + x for x in p['denoising']['readme']]
    d = t['dx_exclusions']
    L += ['\n## TCGA-PRAD\n', '| Step | Count | Source / check |', '|---|---|---|',
          f"| GDC Diagnostic Slides | {f['tcga_manifest']} slides, {f['tcga_manifest_patients']} patients, {t['gdc_manifest']['total_size_gb']} GB, slide types {t['gdc_manifest']['slide_type_codes']}, sample codes {t['gdc_manifest']['sample_codes']} | `gdc_manifest.tsv` |",
          f"| Excluded (label-based, before any image processing) | {d['n_excluded']}: " + '; '.join(f'{k}: {v}' for k, v in d['reasons'].items()) + f" | `clinical.csv` ({t['clinical']['n_cases']} cases) with the notebook's `gleason_to_isup`; {d['n_excluded_patients']} patients, {d['n_excluded_patients_with_other_included_slides']} of whom have another included slide |",
          f"| Analysed cancer cohort | {f['tcga_dx']} DX slides, {f['tcga_dx_patients']} patients, {f['tcga_dx_sites']} sites | reproduces `labels.csv` ids / ISUP: {d['reproduces_labels_csv_ids']} / {d['reproduces_labels_csv_isup']} |",
          f"| Benign Tissue Slides | {f['tcga_ts']} TS slides, {f['tcga_ts_patients']} patients, {f['tcga_ts_sites']} sites, sample codes {t['ts_benign']['sample_codes']} | `labels_benign.csv`; {t['ts_benign']['query']} |",
          f"| Feature files / predictions | DX {t['artefacts']['n_feature_files_dx']} / {t['artefacts']['n_predictions_dx']}, TS {t['artefacts']['n_feature_files_ts']} / {t['artefacts']['n_predictions_ts']}; sets equal to the labelled cohorts: "
          f"{t['artefacts']['dx_features_equal_labelled']}, {t['artefacts']['dx_predictions_equal_labelled']}, {t['artefacts']['ts_features_equal_labelled']}, {t['artefacts']['ts_predictions_equal_labelled']} | `CLAM_TR_Features/TCGA_PRAD/uni_20x(_benign)`, `CLAM_TR_Results/TCGA_PRAD/predictions_20x(_benign).csv` |",
          f"\nDX cohort ISUP (slide level): {json.dumps(t['dx_cohort']['isup_slide_level'])}; slides per patient: {json.dumps(t['dx_cohort']['slides_per_patient'])}; "
          f"patients with conflicting ISUP across slides: {t['dx_cohort']['patients_with_conflicting_isup']} (labels are case-level).",
          f"\nSites (tissue source site code): {f['tcga_dx_sites']}; sites with >= 4 / >= 8 / >= 10 / >= 20 / >= 30 slides: "
          + ' / '.join(str(t['dx_cohort']['sites_with_n_slides_ge'][k]) for k in ('4', '8', '10', '20', '30'))
          + ' (covering ' + ' / '.join(str(t['dx_cohort']['slides_in_sites_with_n_ge'][k]) for k in ('4', '8', '10', '20', '30')) + ' slides); by patients: '
          + ' / '.join(str(t['dx_cohort']['sites_with_n_patients_ge'][k]) for k in ('4', '8', '10', '20', '30')) + '.',
          f"\nSlides per site: {json.dumps(t['dx_cohort']['slides_per_site'])}.",
          f"\nBenign TS patients who also have a DX slide in the 426-slide cohort: {t['ts_benign']['patients_also_in_dx426']} of {t['ts_benign']['n_patients']} (in the 449 manifest: {t['ts_benign']['patients_also_in_manifest449']}).",
          '\n### The 23 excluded Diagnostic Slides\n', '| Slide id | Patient | Reason | Primary | Secondary |', '|---|---|---|---|---|']
    for e in d['excluded_slides']:
        L.append(f"| {e['slide_id']} | {e['patient']} | {e['reason']} | {e['primary']} | {e['secondary']} |")
    L.append(f"\nRuntime: {r['meta']['runtime_s']} s.")
    OUT_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
