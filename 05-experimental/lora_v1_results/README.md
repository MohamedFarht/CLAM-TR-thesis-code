# CLAM-TR LoRA v1: pipeline results

Record of the exploratory LoRA fine-tuning branch of the thesis on PANDA (9,128 slides, the same five-fold split as
the factorial study). The branch followed a pre-registered five-stage protocol (thesis Appendix 1; Zenodo deposit
https://doi.org/10.5281/zenodo.20529009). Stages 0-4 were completed; encoder adaptation gave no clear gain, and the
branch is not used as evidence for any contribution of the thesis (thesis Section 4.9 and Appendix 1.4). This
directory holds the small JSON result files; checkpoints, encoded features and patch archives stay in the Google
Drive workspace of the runs because of their size.

**Pipeline notebook:** `../CLAM_TR_LoRA_v1_FIXED_v8_SHARDED.ipynb` (39 cells, Stages 0-4, A100 80 GB with about
179 GB system RAM). The held-out Stage 3 feature re-extraction is also available as
`../../01-data-preparation/LoRA_Feature_Extraction_FINAL_v4.ipynb`.

Two corrections made during the work are contained in the released notebooks:
- Stage 3 encodes every slide with the LoRA checkpoint of its own held-out fold (slide in fold `N` ->
  `lora_fold{N}_best.pt`), so no slide is encoded by an encoder fine-tuned on it. An earlier single-threaded Stage 3
  cell used one checkpoint for all slides; the released cell 34 is the multi-threaded held-out pipeline ported from
  `LoRA_Feature_Extraction_FINAL_v4.ipynb`.
- The Stage 3 extraction uses 224-pixel patches at 20x, the input size of the LoRA training, and reads the Hugging
  Face token from a Colab secret (otherwise from the placeholder `hf_YOUR_HF_TOKEN_HERE`).

## Five-stage pipeline

| Stage | Description (thesis Section 2.6.2) | Status | Source in the notebook |
|------:|-------------|--------|-----------------|
| 0 | One frozen-feature UNI SpatialMultRetNet_bestHP fold checkpoint (weights of its last saved epoch) scores every patch of every slide; the TopK = 128 patch indices and coordinates are saved (`patch_indices/top128_indices.json`). | complete, 9,128 slides | cell 19 (`run_stage_0()`) |
| 1 | The 128 selected patches are read from each slide's `.tiff` file as 224 × 224-pixel crops at 20x, at the stored coordinates of the selected 256-pixel patches; one `.pt` tensor per slide. | complete, 9,128 slides, 0 failed | cell 22 (`_extract_one_slide()`); `lora_extraction_progress.json` |
| 2 | UNI (ViT-L/16) fine-tuned with LoRA adapters on the `qkv` projections (`r = 8`, `alpha = 16`, dropout 0.05) jointly with a SpatialMultRetNet head matched to the frozen best-HP configuration; up to 10 epochs per fold, patience 5. | complete, 5/5 folds | cell 32 (`run_stage_2()`); `lora_training_results.json` |
| 3 | Every slide re-tiled with 224-pixel patches at 20x and encoded by the LoRA encoder of its held-out fold (5,516,732 patches in total). | complete, 9,128 slides | cell 34; `lora_features/*.pt` on the Drive |
| 4 | The six aggregators × two hyperparameter sets × five folds re-run on the LoRA features, same folds and code. | complete, 12 configurations | cell 36 (`run_stage_4()`); `v20_lora_features_results.json` |

## Files in this directory (9 JSON files)

| File | Size | Description |
|------|-----:|-------------|
| `lora_training_results.json` | 2.6 KB | Stage 2 five-fold summary: per-fold QWK, accuracy, MAE, within-1, best epoch, fold time, per-grade accuracy; aggregate `mean_qwk`, `std_qwk`, `mean_acc`, `std_acc`, `best_fold`. |
| `v20_lora_features_results.json` | 3.1 KB | Stage 4 matrix of the 12 configurations: per-configuration `fold_qwks` (5 values, fold order), `mean_qwk`, `std_qwk`. |
| `v20_fold_progress.json` | 2 B | Stage 4 progress checkpoint (empty `{}` after completion). |
| `lora_fold{0..4}_predictions.json` | 75-81 KB each | Stage 2 per-fold validation predictions per slide: `{slide_ids, predictions, labels}`. Lengths 1803 / 1787 / 1816 / 1933 / 1789, sum 9,128. |
| `lora_extraction_progress.json` | 329 KB | Stage 1 list of completed slide IDs (9,128), empty `failed` list, time stamp `2026-05-14T18:04:04.167049`. |

The `std_*` fields of the JSON files are population standard deviations (ddof = 0). The thesis reports the
across-fold sample standard deviation (ddof = 1), for example 0.9563 ± 0.0030 for Stage 2 and 0.9529 ± 0.0026 for
the best Stage 4 configuration.

## SHA256 hashes

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

## Large artefacts (not in the repository)

Folder `PANDA_LoRA/` in the Google Drive workspace of the runs:

| Artefact | Path under `PANDA_LoRA/` | Size / count |
|----------|------------|-------------:|
| LoRA fold checkpoints | `checkpoints/lora_fold{0..4}_best.pt` | 5 × 19,995,369 B (about 20 MB each) |
| Held-out LoRA features | `lora_features/{slide_id}.pt` | 9,128 files |
| Extracted patches | `extracted_patches/{slide_id}.pt` | 9,129 files (one extra control file) |
| Patch tar archive | `extracted_patches.tar` | 27,179,089,920 B (about 27 GB) |
| Sharded patch tars | `patches_shards/` | 19 shards |
| Patch indices (Stage 0 output) | `patch_indices/top128_indices.json` | 1 file |
| Stage 1 progress checkpoint | `progress/lora_extraction_progress.json` | copy in this directory |

## Stage 2 (LoRA encoder direct classifier), values as stored

```
mean_qwk  = 0.9563308262662703
std_qwk   = 0.002685849734025328   (population SD; sample SD 0.0030 in the thesis)
mean_acc  = 0.826560970613536
std_acc   = 0.007082535563145895
best_fold = 3 (fold-level QWK = 0.9591734298340241)
```

Per-fold metrics (rounded; exact values in `lora_training_results.json`):

| Fold | QWK | Accuracy | MAE | within-1 | best_epoch | fold_time_min |
|----:|-----:|----------:|------:|----------:|-----------:|--------------:|
| 0 | 0.9531 | 0.8242 | 0.2035 | 0.9789 | 9 | 6.18 (see below) |
| 1 | 0.9530 | 0.8260 | 0.1987 | 0.9821 | 10 | 472.33 |
| 2 | 0.9585 | 0.8398 | 0.1817 | 0.9829 | 8 | 472.33 |
| 3 | 0.9592 | 0.8184 | 0.2012 | 0.9834 | 7 | 471.43 |
| 4 | 0.9578 | 0.8245 | 0.1934 | 0.9855 | 10 | 474.45 |

The Stage 2 value is a different metric from the pre-registered Stage 4 evaluation (the encoder and the head are
trained jointly), and for four of the five folds it is not a fully held-out value, because the Stage 0 model that
selected their 128 patches was trained on those slides. The thesis reports it for completeness and does not interpret
it against the decision rule (thesis p.45 and Appendix 1.4).

Fold-0 wall time: the logged `fold_time_min` of fold 0 is 6.18 minutes, whereas folds 1-4 sit at about 472 minutes
each. The fold-0 QWK (0.9531) is consistent with the other folds and its best epoch lies within the 10-epoch budget,
which rules out a silent training failure; the most likely explanation is a session-resume timer reset under the
single-fold-per-session workflow. Treat the fold-0 time as unrecorded; for aggregate wall time use the mean of folds
1-4 (about 472.6 minutes).

## Stage 4 (LoRA features through the six aggregators), values as stored

Sorted by mean QWK; exact values in `v20_lora_features_results.json`; `std_qwk` is the population SD.

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

The default-hyperparameter LoRA runs used a learning rate of 1 × 10^-4, whereas the frozen-feature factorial used
2 × 10^-4 for both sets (thesis p.45).

## Outcome against the decision rule (thesis Appendix 1.2 and 1.4)

Decision rule on the five-fold mean QWK: above 0.960 a clear gain; above 0.955 but below 0.960 a marginal gain;
[0.952, 0.955] within the fold-level noise envelope of the frozen-feature UNI2-h SpatialMultRetNet_bestHP
configuration (0.9530 ± 0.0033, thesis Table 9); below 0.952 no gain.

- The best Stage 4 configuration (SpatialAdd_defaultHP, 0.9529 ± 0.0026) falls in the within-noise bucket. It is
  below the best frozen-UNI configuration (RetNet_bestHP, 0.9543 ± 0.0051).
- Pre-registered Contrast 1 (RetNet_bestHP vs ABMIL_bestHP on UNI) shifts from +0.0087 on the frozen encoder to
  +0.0046 on the LoRA-adapted encoder, a magnitude reduction of approximately 48%, below the 50% threshold of the
  decision rule, so the rule is not triggered. The fold-paired change is −0.0041 (95% CI [−0.0078, −0.0004]).
- Pre-registered Contrast 2 (SpatialMultRetNet_bestHP vs ABMIL_bestHP on UNI2-h) is not evaluable, because LoRA
  fine-tuning of UNI2-h is reserved for future work (thesis Section 6.1.3).
- The planned tests of Appendix Table 1 are reproduced by `06-thesis-figures/analysis/lora_planned_tests.py`, and
  Appendix Figures 1-3 by `06-thesis-figures/plots/make_appendix_figs_1_2_3.py`.

The completed protocol therefore shows no clear gain from encoder adaptation on UNI.

## How to re-run

1. Restore the runtime of the notebook (cell 6: `peft`/`torchao` compatibility fix; cell 7: dependency install) on
   an A100 runtime and make `MahmoodLab/UNI` downloadable (your own token; the notebook shows the placeholder
   `hf_YOUR_HF_TOKEN_HERE`).
2. Cell 11 holds the configuration (paths, LoRA settings, training hyperparameters); cell 15 the v20 model
   components; cell 17 the loss and metrics.
3. Stage 2: cell 32 (about 8 hours per fold on the A100). Stage 3: cell 34 (held-out, multi-threaded). Stage 4:
   cell 36 (12 configurations × 5 folds × up to 50 epochs).
4. Outputs go to `PANDA_LoRA/lora_features/` (Stage 3) and `PANDA_LoRA/results/v20_lora_features_results.json`
   (Stage 4) on the mounted Drive.

## Completion timeline (recorded)

| Date | Event |
|------|-------|
| 2026-05-11 | Stage 0 + Stage 1 complete (patch indices, raw patches) |
| 2026-05-12 | Stage 2 fold 0 complete |
| 2026-05-13 | Stage 2 folds 1, 2 complete |
| 2026-05-14 | Stage 2 folds 3, 4 complete; Stage 3 launched (multi-threaded, held-out) |
| 2026-05-14 18:04 | Stage 3 complete (9,128 features, 0 failures) |
| 2026-05-16 | Stage 4 complete (12 configurations) |
