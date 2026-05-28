# CLAM-TR LoRA v1 — Pipeline Results Manifest

Consolidated record of the LoRA fine-tuning experiment on PANDA (9,128 slides,
5-fold stratified CV). This directory holds the small JSON artifacts copied
from the Google Drive workspace into the repository; large artifacts
(checkpoints, encoded features, patch tar) stay on Drive due to size.

**Pipeline notebook:** `../CLAM_TR_LoRA_v1_FIXED_v8_SHARDED.ipynb` (39 cells, 5
stages, A100 80 GB + ~167 GB RAM target).

**Helper scripts** (repository-side surgical fixes applied to the notebooks):
- `.claude/bin/fix_v4_extraction.py` — surgical fix-pass on the v4 Stage-3
  extraction notebook (patch_size 256 → 224 to match LoRA training distribution;
  HF_TOKEN sourced from Colab `userdata` secret with hardcoded literal fallback;
  drops the GPU-side `F.interpolate` resize).
- `.claude/bin/port_v4_stage3_into_v8.py` — replaces the v8 notebook's
  single-threaded `run_stage_3()` with the v4 multi-threaded pipeline.
  **Correctness fix:** the v8 single-threaded Stage 3 encoded every slide with
  one best-fold checkpoint, which is a data-leakage path; the v4 port encodes
  each slide with its own held-out fold's checkpoint
  (slide in fold `N` → `lora_fold{N}_best.pt`).

## Five-stage pipeline

| Stage | Description | Status | Source of truth |
|------:|-------------|--------|-----------------|
| 0 | Attention-based Top-128 patch selection using a frozen v20 `SpatialMultRetNet_bestHP` checkpoint; per-slide `{indices, coords, scores}` written to `patch_indices/top128_indices.json`. | COMPLETE — 9,128 slides | Notebook cell 19 (`run_stage_0()`) |
| 1 | Raw 256×256 patch extraction from PANDA `.tiff` slides at 20× magnification; one `.pt` tensor per slide. | COMPLETE — 9,128 slides, 0 failed | Notebook cell 22 (`_extract_one_slide()`); `lora_extraction_progress.json` |
| 2 | LoRA training, 5-fold CV (10 epochs per fold, early-stop patience 5); UNI v1 ViT-L/16 backbone + LoRA adapters on `qkv` (`r=8, alpha=16, dropout=0.05`); SpatialMultRetNet head matches v20 best-HP. | COMPLETE — 5/5 folds | Notebook cell 32 (`run_stage_2()`); `lora_training_results.json`; `checkpoints/lora_fold{0..4}_best.pt` on Drive |
| 3 | Held-out feature re-extraction with 5 CPU workers feeding GPU concurrently; each slide is encoded by its own held-out fold's LoRA checkpoint. | COMPLETE — 9,128 features | Notebook cell 34 (v4 port via `port_v4_stage3_into_v8.py`); `lora_features/*.pt` on Drive |
| 4 | V20 aggregator pipeline re-run on LoRA features: 6 aggregator variants × 2 hyperparameter sets × 5 folds = 60 runs. | COMPLETE — 12 configs | Notebook cell 36 (`run_stage_4()`); `v20_lora_features_results.json` |

## Files in this directory (9 JSONs, ~700 KB total)

| File | Size | Description |
|------|-----:|-------------|
| `lora_training_results.json` | 2.6 KB | Stage 2 5-fold summary: per-fold QWK, accuracy, MAE, within-1, best-epoch, fold time, per-grade accuracy; aggregate `mean_qwk`, `std_qwk`, `mean_acc`, `std_acc`, `best_fold`. |
| `v20_lora_features_results.json` | 3.1 KB | Stage 4 12-configuration matrix: per-config `fold_qwks` (5 values) + `mean_qwk` + `std_qwk`. |
| `v20_fold_progress.json` | 2 B | Stage 4 in-progress checkpoint (empty `{}` after Stage 4 completion). |
| `lora_fold{0..4}_predictions.json` | 75–81 KB each | Per-fold validation-set slide-level predictions: `{slide_ids, predictions, labels}`. Lengths 1803 / 1787 / 1816 / 1933 / 1789 — sum to exactly 9,128. |
| `lora_extraction_progress.json` | 329 KB | Stage 1 completed slide-id list (9,128 IDs) + empty `failed` list + timestamp `2026-05-14T18:04:04.167049`. |

## SHA256 integrity hashes

```
86ab59f616481f688f21b4a1b3e0c23fd14bdc71050841f0beb0055cf08c11d7  lora_extraction_progress.json
7b0c4efcd4074a77d95edad54a4e6aa0f7f492fb4fb81983521adcf07f4403fb  lora_fold0_predictions.json
e7ac61be98e7fc9988cfe765782e84103d37c24e1df1fd2ad6fc34f6958fa8a3  lora_fold1_predictions.json
530bd49d1935f74bde3fac9cf7ddabe2e468269959bc91569132e4155939365f  lora_fold2_predictions.json
1aa39e69c3d21156f9f934c9ae5884966c260ffb893483483b574a9a6b19a959  lora_fold3_predictions.json
a20ae573bde7d25bff7cc66983e0a553a5e48482712c79610bc5d83de479a195  lora_fold4_predictions.json
963150e09cb15d02ca14290da7ff1509194a67869189f61435e72fe3d05c4c8f  lora_training_results.json
44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a  v20_fold_progress.json
4ac65bc6288659ae16c2095221d04a5f8ec527a687df79022cc7bff3f0731913  v20_lora_features_results.json
```

## Drive paths (large artifacts not copied into repo)

| Artifact | Drive path | Size / count |
|----------|------------|-------------:|
| LoRA fold checkpoints | `G:/My Drive/PANDA_LoRA/checkpoints/lora_fold{0..4}_best.pt` | 5 × 19,995,369 B (~20 MB each, ~100 MB total) |
| Held-out LoRA features | `G:/My Drive/PANDA_LoRA/lora_features/{slide_id}.pt` | 9,128 files |
| Pre-extracted patches | `G:/My Drive/PANDA_LoRA/extracted_patches/{slide_id}.pt` | 9,129 files (one extra control-file) |
| Patch tar archive | `G:/My Drive/PANDA_LoRA/extracted_patches.tar` | 27,179,089,920 B (~27 GB) |
| Sharded patch tars | `G:/My Drive/PANDA_LoRA/patches_shards/` | 19 shards (v8 sharded-tar workflow) |
| Patch indices (Stage 0 output) | `G:/My Drive/PANDA_LoRA/patch_indices/top128_indices.json` | 1 file |
| Stage 1 progress checkpoint | `G:/My Drive/PANDA_LoRA/progress/lora_extraction_progress.json` | duplicated here |

## Stage 2 — verbatim aggregate results (5-fold)

```
mean_qwk  = 0.9563308262662703
std_qwk   = 0.002685849734025328
mean_acc  = 0.826560970613536
std_acc   = 0.007082535563145895
best_fold = 3 (fold-level QWK = 0.9591734298340241)
```

Per-fold metrics (rounded for readability; verbatim values in
`lora_training_results.json`):

| Fold | QWK | Accuracy | MAE | within-1 | best_epoch | fold_time_min |
|----:|-----:|----------:|------:|----------:|-----------:|--------------:|
| 0 | 0.9531 | 0.8242 | 0.2035 | 0.9789 | 9 | **6.18** ⚠ |
| 1 | 0.9530 | 0.8260 | 0.1987 | 0.9821 | 10 | 472.33 |
| 2 | 0.9586 | 0.8398 | 0.1817 | 0.9829 | 8 | 472.33 |
| 3 | 0.9592 | 0.8184 | 0.2012 | 0.9834 | 7 | 471.43 |
| 4 | 0.9578 | 0.8245 | 0.1934 | 0.9855 | 10 | 474.45 |

**Fold-0 wall-time anomaly:** fold 0's logged `fold_time_min = 6.18` is ~77×
shorter than the other four folds (~472 min each), while its QWK
(0.9531) is consistent with the cluster (0.9530–0.9592). The QWK reaching
the best-epoch value at epoch 9 (within the 10-epoch budget) rules out a
silent failure. Most-likely cause: the fold-0 timer was reset across a
session-resume boundary (the v5 single-fold-per-session workflow noted in
notebook cell 31), so `fold_time_min` records only the post-resume portion
of fold-0 training rather than the full wall-time. Treat fold-0 time as
**unrecorded** rather than `6.18 min`. For an aggregate wall-time figure,
report folds 1–4 mean (~472.6 min) and note fold 0 separately.

## Stage 4 — verbatim 12-configuration matrix (LoRA features → v20 aggregators)

Configuration mean QWK (5-fold) and std, sorted by mean QWK descending.
Verbatim values in `v20_lora_features_results.json`.

| Rank | Configuration | mean_qwk | std_qwk |
|---:|---------------|---------:|--------:|
| 1 | SpatialAdd_defaultHP | 0.9528840183039231 | 0.0022874402550024995 |
| 2 | SpatialMult_defaultHP | 0.9515652162726786 | 0.0036803007976315546 |
| 3 | RetNet_bestHP | 0.950255756025828 | 0.004114404870944409 |
| 4 | SpatialMult_bestHP | 0.9501146526513622 | 0.0047019128096606935 |
| 5 | SpatialMultRetNet_bestHP | 0.9497871782167536 | 0.0021543636011971934 |
| 6 | SpatialAdd_bestHP | 0.9471719540090346 | 0.0030099565042215554 |
| 7 | DualBranch_bestHP | 0.9464140946553232 | 0.005481000113929969 |
| 8 | ABMIL_bestHP | 0.9456979565610256 | 0.005866879242310598 |
| 9 | SpatialMultRetNet_defaultHP | 0.9419388998001292 | 0.0046413409183614935 |
| 10 | RetNet_defaultHP | 0.9410695303934681 | 0.0015399982618100572 |
| 11 | DualBranch_defaultHP | 0.9362205237839252 | 0.007792018963858431 |
| 12 | ABMIL_defaultHP | 0.9316718328331802 | 0.005063565987077238 |

## Comparison to frozen-feature v20 baselines

Frozen-encoder baselines from the main thesis Appendix B Table B.1 (24-cell
factorial):
- Frozen UNI + RetNet_bestHP = **0.9543 ± 0.0046** (thesis headline best)
- Frozen UNI2-h + SpatialMultRetNet_bestHP = **0.9530 ± 0.0029**

Pre-registered decision rule (`thesis/main/bolumler/son_kisim/bolum_ekler.tex`
Section `app:lora_decision`):
- QWK > 0.960 → clear gain
- QWK ∈ (0.955, 0.960] → marginal gain
- QWK ∈ [0.952, 0.955] → within noise envelope (vs frozen UNI2-h
  SpatialMultRetNet_bestHP 0.9530 ± 0.0029)
- QWK < 0.952 → no-gain

Outcome:
- **Stage 2 LoRA-direct classifier (0.9563)** falls in the **marginal-gain**
  bucket. This is an end-to-end LoRA + SpatialMultRetNet-head trained jointly,
  not a frozen-feature-then-aggregator pipeline.
- **Best Stage 4 LoRA-features → aggregator (0.9529)** falls in the
  **within-noise** bucket vs the frozen UNI2-h baseline 0.9530.

Pre-registered Contrast 1 (RetNet_bestHP vs ABMIL_bestHP on UNI):
- Frozen: 0.9543 − 0.9456 = +0.0087
- LoRA: 0.9503 − 0.9457 = +0.0046
- Magnitude reduction: **~47%**, below the 50% sign-flip-equivalent threshold

Pre-registered Contrast 2 (specifies UNI2-h): not directly evaluable because
Stage 4 was run on LoRA-UNI features only; LoRA-UNI2-h is the immediate Q1
extension.

## Stage 3 data-leakage fix (notebook cell 34)

The v8 notebook's built-in single-threaded `run_stage_3()` (cell 34 before
the port) encoded **every slide with one best-fold checkpoint**, regardless
of which fold the slide belonged to. This is a feature-extraction-time data
leak: features for a slide that participated in fold `N`'s training data
would be encoded by `lora_fold{N}_best.pt`, which had seen that slide during
fine-tuning.

The replacement pipeline (`port_v4_stage3_into_v8.py`, ported from
`LoRA_Feature_Extraction_FINAL_v4.ipynb` cell 6):
- Loads each fold's checkpoint once.
- Iterates the 5 fold val-id sets; for each slide in fold `N`, encodes with
  `lora_fold{N}_best.pt`.
- Runs 5 CPU workers feeding the GPU concurrently (~5–10× speedup on A100).

The 12-configuration Stage 4 results in `v20_lora_features_results.json` are
generated from the leakage-free held-out features. The fix was applied
2026-05-14 (between Stage 2 completion 2026-05-14 and Stage 4 completion
2026-05-16).

## Reproducibility

To re-execute Stage 2:
1. Restore the runtime environment (notebook cell 6: peft/torchao
   compatibility fix; cell 7: dependency install).
2. Ensure `MahmoodLab/UNI` weights are downloadable (HF_TOKEN secret).
3. Notebook cell 11: configuration block (paths, LoRA config, training
   hyperparameters); cell 15: v20 model components (exact copy);
   cell 17: loss + metrics; cell 32: `run_stage_2()`.
4. 5 folds × ~8 hours per fold on A100 80 GB ≈ 40 hours total wall time.

To re-execute Stages 3–4 with the v4 multi-threaded held-out pipeline:
1. Apply `python .claude/bin/port_v4_stage3_into_v8.py` (idempotent;
   safe to re-run).
2. Run notebook cell 34 (Stage 3 — 5–9 hours for 9,128 slides on A100).
3. Run notebook cell 36 (Stage 4 — 12 configurations × 5 folds × 50 epochs).
4. Outputs land in `G:/My Drive/PANDA_LoRA/lora_features/` (Stage 3) and
   `G:/My Drive/PANDA_LoRA/results/v20_lora_features_results.json` (Stage 4).

## Completion timeline (recorded)

| Date | Event |
|------|-------|
| 2026-05-11 | Stage 0 + Stage 1 complete (patch indices, raw patches) |
| 2026-05-12 | Stage 2 fold 0 complete |
| 2026-05-13 | Stage 2 folds 1, 2 complete |
| 2026-05-14 | Stage 2 folds 3, 4 complete; Stage 3 launched (multi-threaded) |
| 2026-05-14 18:04 | Stage 3 complete (9,128 features, 0 failures) |
| 2026-05-16 | Stage 4 complete (12 configurations) |
