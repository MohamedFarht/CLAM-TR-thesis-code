# Tables and figures of the thesis: generators and data sources

Thesis: *Attention-Enhanced Dual-Branch Aggregation for Whole Slide Image Classification in Prostate Cancer
Grading*, Master's thesis, Karadeniz Technical University, 2026. Page numbers are the printed page numbers of the
final thesis. Every table and figure is listed once, with the script in this folder that produces it and the file
the numbers come from.

## 1. Folder contents

| Folder | Contents |
|---|---|
| `analysis/` | statistics scripts and the calibration notebook |
| `plots/` | figure scripts (matplotlib) |
| `tikz/` | TikZ sources of the drawn figures, compiled standalone (see `tikz/README.md`) |
| `data/` | small derived input files (section 5) |
| `CLAM_TR_Thesis_Figures*.ipynb` | early figure notebooks of April 2026, superseded (section 7) |

## 2. How to run the scripts

Run from the repository root with `python -P <script>` (for example
`python -P 06-thesis-figures/analysis/fold_aligned_recompute.py`). Packages: `requirements.txt`. The verification
runs of 16 Nov 2026 used Python 3.13.3, numpy 2.2.6, matplotlib 3.10.8 and seaborn 0.13.2, with scipy, pandas,
scikit-learn, statsmodels, PyMuPDF and Pillow. Each script writes its output next to itself, as during the thesis
work. Each script starts with a header that names the thesis output, the inputs and the run command. The computations
are those used for the thesis; lines marked `# repo path` read paths inside this repository or the environment
variables below, and `lora_planned_tests.py` skips the checks whose inputs are not in the repository (note A).

Inputs that are not in the repository are found through these environment variables (all optional):

| Variable | Meaning | Default |
|---|---|---|
| `CLAM_TR_V20_DIR` | results folder of the factorial runs, `CLAM_TR_Results/v20_fullscale`, as written by the notebooks in `04-final-v20/`: `<encoder>/v20_final_summary.json`, `<encoder>/predictions/`, `<encoder>/predictions_attention/`, `<encoder>/representative_slides.json` | `06-thesis-figures/data/v20_fullscale` (holds only the two summary files) |
| `CLAM_TR_PROJECT_DIR` | folder with inputs that are not redistributed, in the layout of the original project: `PANDA/dataset/train.csv`, `Kaggle-PANDA-1st-place-solution/Kaggle-PANDA-1st-place-solution-master/input/`, and the thesis build that the consistency checks compare with (`thesis/main/bolumler/...` LaTeX sources, `thesis/main/figures/*.pdf`) | repository root |
| `CLAM_TR_DRIVE_DIR` | Google Drive root of the Colab runs (only for the TCGA-PRAD part of `cohort_flow_counts.py`) | `/content/drive/MyDrive` |
| `CLAM_TR_ATTENTION_STATS_DIR` | folder with the per-slide output of `analysis/attention_fold0_statistics.py` (input of Figure 15) | `06-thesis-figures/analysis` |
| `CLAM_TR_FOLD_FACTS` | optional per-fold check file of Figure 5 (its gate 4 is skipped without it) | `06-thesis-figures/data/fold_facts.json` (not shipped) |
| `CLAM_TR_FOLD_QWK_COPY` | optional independent copy of the fold-order QWKs, `r2c1_c2_results.json` (sibling repository CLAM-TR); a cross-check of `lora_planned_tests.py`, skipped without it | `06-thesis-figures/data/r2c1_c2_results.json` (not shipped) |
| `CLAM_TR_LORA_CHECK_LOG` | optional log of an earlier check of the LoRA results, `lora_checks_out.txt`; a cross-check of `lora_planned_tests.py`, skipped without it | `06-thesis-figures/data/lora_checks_out.txt` (not shipped) |

Four details of the data:

1. **Fold pairing.** The per-fold QWK lists stored in `v20_final_summary.json` are in the order in which the folds
   finished, which differs between configurations. Every paired statistic here pairs the folds by fold index, using
   the fold number in the name of each per-slide prediction file (thesis p.37 and p.54). The earlier analysis that
   paired the lists by position is superseded.
2. **UNI2-h ABMIL_bestHP, fold 0.** The thesis uses the original training run (fold QWK 0.9500, five-fold mean
   0.9433). A later retraining of that fold with the same seed (QWK 0.9476, mean 0.9428) is used by the companion
   journal paper (thesis p.49). To reproduce the thesis values, use a results folder that holds the original file;
   `fold_aligned_recompute.py` and `verify_chapter3_numbers.py` also print the statistics with the retrained value.
3. **UNI ABMIL_defaultHP, fold 4.** The results folder holds two prediction files for this fold
   (`ABMIL_defaultHP_fold4_predictions.json` and `ABMIL_defaultHP_fold4_predictions (1).json`). The scripts that match
   each file against the stored fold values use the second one, which reproduces the stored summary.
   `full_factorial_24_runs.py` reads the first one and therefore differs from Appendix Table 2 on this row only.
4. **Standard deviations.** The v20 notebooks store and print the population SD of the five fold values (ddof = 0),
   for example an SD of 0.0046 for UNI RetNet_bestHP (mean 0.9543). The thesis reports the across-fold sample SD
   (ddof = 1), 0.9543 ± 0.0051 (Table 9), as do the scripts of `analysis/`. The `std_*` fields of the LoRA result
   files are also population SDs.

The plotting scripts re-set the figures for print and end with a check against the figure file of the thesis build
(same words and numbers, printed text of at least 10 pt, no overlapping text). They therefore need the thesis build
under `CLAM_TR_PROJECT_DIR`. The scripts of Figures 4, 9 and 14 take the image rasters (slide thumbnails and patch
crops, which come from the PANDA slides and are not redistributed) from the previous version of the figure file.

## 3. Tables

Status: **reproduced** = script re-run on 16 Nov 2026 and every printed cell equal (cells compared by script with the
thesis text); **documented** = generator and source named, not re-run (reason given); **literature or text** = no
script, the content comes from the cited sources or from the thesis text.

| Table | p. | Title as printed | Kind | Generator | Data source | Status |
|---|---|---|---|---|---|---|
| 1 | 3 | Inter-pathologist agreement on the PANDA validation cohorts and the upstream Karolinska and Radboud studies | literature | none | cited studies | literature |
| 2 | 8 | Pathology foundation-model encoders | literature | none | cited studies | literature |
| 3 | 10 | MIL aggregators surveyed across the WSI literature | literature | none | cited studies | literature |
| 4 | 18 | Slide distribution of the 9,128-slide noise-cleaned PANDA subset across ISUP grade groups | data | `analysis/cohort_flow_counts.py` | `data/cohort_flow_counts.json` (`panda.cohort_9128.isup`), computed from the noise-cleaned fold file of the 1st-place release | reproduced |
| 5 | 19 | PANDA cohort provenance and reference-standard disclosure by contributing institution (Bulten et al., 2022; Team PND et al., 2020) | data and literature | `analysis/cohort_flow_counts.py` (slide counts) | `data/cohort_flow_counts.json` (`panda.release`, `panda.denoising`, `panda.cohort_9128`); reference standards from the cited papers | reproduced (counts) |
| 6 | 38 | Six aggregator variants of the factorial study | text | none | model classes of the v20 notebooks (`04-final-v20/`) | text |
| 7 | 38 | Aggregator parameter counts, UNI (1024-d) and UNI2-h (1536-d) encoders | data | `analysis/recompute_summary_and_appendix_numbers.py`, `analysis/verify_chapter3_numbers.py` | `params` in `data/v20_fullscale/*/v20_final_summary.json` | reproduced |
| 8 | 40 | Hyper-parameter sets used in the factorial study | text (configuration) | none | `HP_SETS` cell of the v20 notebooks; the same values are stored as `hp_sets` in `data/v20_fullscale/*/v20_final_summary.json` | text |
| 9 | 49 | Main bestHP factorial results on the 9,128-slide noise-cleaned PANDA subset | data | `analysis/verify_chapter3_numbers.py`; SDs also `analysis/sd_ddof1_and_t_intervals.py` | summary files + per-slide prediction files; linear-probe row: printed output of the v20 notebooks ("Linear probe QWK = 0.8121" in the UNI notebooks, "0.8414" in the UNI2-h notebooks) | reproduced |
| 10 | 50 | Parametric Student-t 95% CIs on QWK for the 12 bestHP factorial configurations | data | `analysis/sd_ddof1_and_t_intervals.py`, `analysis/verify_chapter3_numbers.py` | per-slide prediction files (fold QWKs) | reproduced |
| 11 | 53 | Factorial decomposition of the four pre-specified contrasts per encoder | data | `analysis/fold_aligned_contrasts_figs12_17_18.py` (writes `factorial_ci_table.csv`), `analysis/fold_aligned_recompute.py` | per-slide prediction files, folds paired by index | reproduced |
| 12 | 54 | UNI encoder: eleven pre-specified pairwise comparisons on fold-level QWK differences | data | `analysis/fold_aligned_recompute.py`, `analysis/verify_chapter3_numbers.py` | per-slide prediction files | reproduced |
| 13 | 55 | UNI2-h encoder: eleven pre-specified pairwise comparisons on fold-level QWK differences | data | `analysis/fold_aligned_recompute.py`, `analysis/verify_chapter3_numbers.py` | per-slide prediction files (original UNI2-h ABMIL fold-0 run, section 2) | reproduced |
| 14 | 58 | Per-grade accuracy for the twelve bestHP configurations of the 24-run factorial | data | `analysis/verify_chapter3_numbers.py` | `fold_metrics[].per_grade_accuracy` in the summary files | reproduced |
| 15 | 58 | Per-class kappa, precision, recall and F1 for the headline UNI RetNet configuration | data | `analysis/verify_chapter3_numbers.py` (pooled block) | per-slide prediction files of UNI RetNet_bestHP, pooled over the five folds | reproduced |
| 16 | 60 | Pooled 6× 6 confusion matrix for the headline UNI RetNet configuration | data | `analysis/verify_chapter3_numbers.py` | as Table 15 | reproduced |
| 17 | 60 | Provider-stratified performance of the headline configuration across the two PANDA centres | data | `analysis/verify_chapter3_numbers.py` (provider block) | as Table 15, plus `data_provider` of the PANDA label file `train.csv` | reproduced |
| 18 | 61 | Cross-encoder rank comparison | data | `analysis/verify_chapter3_numbers.py` (ranks block) | bestHP means of Table 9 | reproduced |
| 19 | 63 | Per-variant training dynamics summary | data | `analysis/verify_chapter3_numbers.py` (time, best epoch mean ± SD) | `total_time_min` and `fold_metrics[].best_epoch` in the summary files | reproduced |
| 20 | 74 | Comparison of CLAM-TR with prior PANDA-evaluated methods on the cohort-version axis | literature | none | cited studies; CLAM-TR row from Table 9 | literature |
| 21 | 91 | Acknowledged limitations of CLAM-TR with forward-pointers to Chapter 6 future work | text | none | thesis text | text |
| 22 | 102 | External-validation prostate cohorts proposed for the post-thesis programme | literature | none | cited studies | literature |
| 23 | 103 | Foundation-model encoder candidates for the post-thesis cross-family factorial | literature | none | cited studies | literature |
| 24 | 107 | Aggregator-extension targets for the post-thesis future-work programme | literature | none | cited studies | literature |
| Appendix 1 | 119 | Planned Stage 4 tests of the LoRA protocol, computed afterwards on the existing folds | data | `analysis/lora_planned_tests.py` (writes the LaTeX table and `lora_planned_tests_results.json`) | `05-experimental/lora_v1_results/*.json` and the LoRA notebook (in this repository); per-slide prediction files of frozen UNI | reproduced (see note A) |
| Appendix 2 | 126 | Full 24-run factorial results on the 9,128-slide noise-cleaned PANDA subset | data | `analysis/recompute_summary_and_appendix_numbers.py` | summary files + per-slide prediction files | reproduced (24/24 rows; `full_factorial_24_runs.py` 23/24, section 2, point 3) |
| Appendix 3 | 126 | Per-grade accuracy with per-fold standard deviations | data | `analysis/fold_aligned_recompute.py` (per-grade block), `analysis/recompute_summary_and_appendix_numbers.py` | `fold_metrics[].per_grade_accuracy` in the summary files; SD = across-fold sample SD | reproduced (72/72 cells) |

Note A. `lora_planned_tests.py` first checks its inputs (166 checks when every optional input is present) and stops
before computing any new test if a check fails. Some checks look up the computed values in the thesis LaTeX sources,
and two groups compare with files that are not in this repository (`CLAM_TR_FOLD_QWK_COPY`,
`CLAM_TR_LORA_CHECK_LOG`, section 2). When these inputs are absent, as in this repository, the script skips those
checks with a printed note. Run from the repository with `CLAM_TR_V20_DIR` set (16 Nov 2026), it passed 93 of 93
checks, ran to the end, and wrote results equal to the shipped `data/lora_planned_tests_results.json` in every result
section and the LaTeX table of Appendix Table 1 (every row occurs verbatim in the thesis appendix source). With the
thesis sources present, three sentence checks fail because those sentences were reworded after the script was written
(the numbers did not change), and the script stops at 163 of 166 checks.

## 4. Figures

Status: **reproduced** = script re-run on 16 Nov 2026 and the PDF it writes is byte-identical to the file in the
thesis build (TikZ: compiles standalone, same page size and words as the thesis file).

| Figure | p. | Title as printed | Kind | Generator | Data source | Status |
|---|---|---|---|---|---|---|
| 1 | 1 | Inverted-triangle problem-statement schematic for prostate cancer Gleason grading | diagram | `tikz/fig_1_Y_inverted_triangle.tex` | none (text of Chapter 1) | reproduced |
| 2 | 4 | Foundation-model encoder and WSI-MIL aggregator lineage timeline 2018-2026 | diagram (literature) | `plots/make_fig2_lineage_timeline.py` with `plots/lineage_timeline_base.py` and `plots/lineage_timeline_previous_version.py` (run the base script first) | box texts written in the scripts (cited literature) | reproduced |
| 3 | 20 | Cohort flow from raw PANDA to the 9,128-slide 5-fold CV used in this thesis | diagram with counts | `tikz/fig_3_cohort_flow.tex` (drawn inline in the thesis source; standalone copy) | counts as in `data/cohort_flow_counts.json` | reproduced (words identical to the printed figure) |
| 4 | 20 | Example whole slide image (WSI) thumbnails for each ISUP grade group from the PANDA release (Bulten et al., 2022) | data (images) | `plots/make_fig4_isup_samples.py` | slide IDs: `representative_slides.json` (results folder, UNI2-h); thumbnails rendered from the slides with OpenSlide, taken from the previous figure file | reproduced |
| 5 | 22 | Five-fold cross-validation split schematic for the 9,128-slide noise-cleaned PANDA cohort | diagram with counts | `plots/make_fig5_cv_split_schematic.py` (output file name `fig_2_3_patient_split_schematic.pdf` is a legacy name; the split is slide-level, Section 2.1.3) | `data/cohort_flow_counts.json` | reproduced |
| 6 | 27 | Hybrid-head design-precedent analogy between Hymba, Samba and CLAM-TR | diagram | `tikz/fig_2_5_hybrid_precedent.tex` | none | reproduced |
| 7 | 29 | CLAM-TR architecture overview | diagram | `tikz/fig_2_1_clam_tr_architecture.tex` | none (drawn from the model code) | reproduced |
| 8 | 30 | Data preprocessing and feature-caching pipeline | diagram | `tikz/fig_2_2_preprocessing_flow.tex` | none | reproduced |
| 9 | 33 | TopK=128 patch-selection visualisation on an illustrative grade-5 slide | data | `plots/make_fig9_topk_selection.py` | fold-0 attention dump of UNI2-h SpatialMultRetNet_bestHP (`predictions_attention/SpatialMultRetNet_bestHP_fold0_attention.json`, saved TopK indices, softmax weights, coordinates); thumbnail from the previous figure file | reproduced (from the previous figure file and the dump) |
| 10 | 36 | CLAM-TR retention-branch sub-component diagram | diagram | `tikz/fig_2_4_retention_branch_detail.tex` | none (drawn from the model code) | reproduced |
| 11 | 46 | Exploratory LoRA pipeline, stages 0-4 (completed pre-registered protocol) | diagram | `tikz/fig_2_6_lora_v1_pipeline.tex` | none | reproduced |
| 12 | 57 | Forest plot of the four-contrast factorial decomposition per encoder | data | `plots/make_figs_12_17_18.py` (data from `analysis/fold_aligned_contrasts_figs12_17_18.py`, imported) | per-slide prediction files, folds paired by index | reproduced |
| 13 | 63 | Distribution of best epochs per aggregator variant | data | `plots/make_fig13_epoch_distribution.py` | best epochs in `data/v20_fullscale/*/v20_final_summary.json`; `data/epoch_distribution_stats.csv` (recomputed by the script and asserted equal) | reproduced |
| 14 | 65 | Top-9 highest-attention patches across tail-case ISUP grade-1, 3, and 5 slides | data (images) | `plots/make_fig14_top_patches.py` | the nine highest-attention patches of each slide (ranked by the fold-0 attention dump of UNI2-h SpatialMultRetNet_bestHP, read from the slides), taken with their labels from the previous version of the three figure files | reproduced (from the previous figure files) |
| 15 | 66 | Failure-mode analysis: per-slide rank-1 attention mass by correctness and true grade | data | `plots/make_fig15_failure_mode_rank1.py` | `attention_fold0_per_slide.csv`, `attention_fold0_results.json` written by `analysis/attention_fold0_statistics.py` from the fold-0 attention dumps | reproduced (plot); the statistics script was not re-run (section 6) |
| 16 | 68 | Reliability diagram for the UNI2-h encoder-best configuration | data | `plots/make_fig16_calibration_reliability.py` | `data/calibration_results.json`, written by cell 59 of `analysis/CLAM_TR_v20_calibration.ipynb` (Colab, GPU; reloads the best-epoch fold checkpoints) | reproduced (plot); the notebook was not re-run |
| 17 | 82 | Normal-approximation 95% CI projection for the SpatialMultRetNet vs ABMIL contrast on UNI2-h | data | `plots/make_figs_12_17_18.py` | as Figure 12 | reproduced |
| 18 | 83 | Effect-size and bootstrap 95% CI for two model-vs-ABMIL differences at 600 and 9,128 slides | data | `plots/make_figs_12_17_18.py` | as Figure 12, plus `data/v16_validation/*.json` (600-slide development cohort) | reproduced |
| 19 | 94 | Single-page synthesis of the CLAM-TR thesis contribution | diagram | `tikz/fig_5_1_contributions_overview.tex` | none | reproduced |
| 20 | 99 | Three-tier future-work roadmap mapping each limitation to its remedy strand | diagram | `tikz/fig_6_2_future_roadmap.tex` | none | reproduced |
| Appendix 1 | 122 | Stage 4 5-fold mean QWK across the twelve LoRA-UNI feature configurations | data | `plots/make_appendix_figs_1_2_3.py` | `05-experimental/lora_v1_results/v20_lora_features_results.json`; per-fold values of the frozen UNI2-h SpatialMultRetNet_bestHP baseline from `data/lora_planned_tests_results.json` | reproduced |
| Appendix 2 | 123 | Frozen UNI vs LoRA-UNI Stage 4 paired comparison across the six aggregator variants | data | `plots/make_appendix_figs_1_2_3.py` | LoRA results + `data/lora_planned_tests_results.json` | reproduced |
| Appendix 3 | 124 | Stage 2 LoRA encoder direct classifier 5-fold validation QWK | data | `plots/make_appendix_figs_1_2_3.py` | `05-experimental/lora_v1_results/lora_training_results.json` | reproduced |

## 5. Files in `data/`

| File | Written by | Used by |
|---|---|---|
| `v20_fullscale/UNI/v20_final_summary.json` | the UNI runs of `04-final-v20/CLAM_TR_v20_UNI_Colab.ipynb` and `CLAM_TR_v20_UNI_Colab_OPTIMIZED.ipynb` (same results folder) | most scripts: fold values (completion order), accuracies, per-grade metrics, best epochs, parameter counts, times, `hp_sets`, `lr_sweep` |
| `v20_fullscale/UNI2-h/v20_final_summary.json` | `04-final-v20/CLAM_TR_v20_UNI2h_Colab.ipynb` (eleven configurations) and `CLAM_TR_v20_SpatialAdd_bestHP.ipynb` (SpatialAdd_bestHP) | as above |
| `v16_validation/v16_all_results.json`, `v16_final_summary.json` | `03-optimization/CLAM_TR_v16_Validation_Experiments.ipynb` (600-slide development cohort) | Figure 18 |
| `cohort_flow_counts.json` | `analysis/cohort_flow_counts.py` (its TCGA-PRAD section belongs to the companion journal paper) | Tables 4-5, Figures 3 and 5 |
| `epoch_distribution_stats.csv` | a cell of the Colab copy of the April figures notebook; recomputed and asserted by `plots/make_fig13_epoch_distribution.py` | Figure 13, Table 19 |
| `calibration_results.json` | `analysis/CLAM_TR_v20_calibration.ipynb`, cell 59 | Figure 16, Section 3.7 |
| `lora_planned_tests_results.json` | `analysis/lora_planned_tests.py` | Appendix Table 1, Appendix Figures 1-2 |
| `factorial_ci_inline.csv` | percentile-bootstrap CIs per configuration from the sibling repository CLAM-TR (`scripts/q1_bootstrap_ci_factorial.py`) | `analysis/verify_chapter3_numbers.py` (comparison with Table 10, p.50) |

File-path strings that recorded the author's local folders inside two of these files were replaced by `<project>` or
`<Drive>`, and the `meta.script` field of `cohort_flow_counts.json` names the script of this repository; no value
changed.

## 6. Inputs that are not in the repository

| Input | Written by | Location in a run's results | Needed for |
|---|---|---|---|
| per-slide prediction files `{agg}_{bestHP,defaultHP}_fold{k}_predictions.json` (`slide_ids`, `labels`, `predictions`) | the v20 notebooks (`04-final-v20/`) | `CLAM_TR_Results/v20_fullscale/<encoder>/predictions/` | Tables 9-19, Appendix Tables 1-3, Figures 12, 17, 18 |
| PANDA label file `train.csv` | PANDA release (Kaggle) | `PANDA/dataset/` | Tables 4, 5, 17 |
| fold files of the 1st-place release (`train-5kfold.csv`, `train-5kfold_remove_noisy_by_0622_rad_13_08_ka_15_10.csv`, `duplicate_imgids_imghash_thres_090.csv`) | Team PND public release | `Kaggle-PANDA-1st-place-solution/Kaggle-PANDA-1st-place-solution-master/input/` | Tables 4, 5; Figures 3, 5 |
| fold-0 attention dumps `{agg}_bestHP_fold0_attention.json` and `representative_slides.json` | an attention-dump cell appended after training to the Colab copy of `CLAM_TR_v20_FullScale_Colab.ipynb` (not part of the released notebook); it reloads the selected-epoch weights at the configuration's attention temperature (thesis p.64) | `CLAM_TR_Results/v20_fullscale/<encoder>/predictions_attention/` and `.../UNI2-h/` | Figures 4, 9, 14, 15; Section 3.6 |
| per-slide fold-0 attention statistics `attention_fold0_per_slide.csv`, `attention_fold0_results.json` | `analysis/attention_fold0_statistics.py` | next to the script | Figure 15; Section 3.6 (p.64) |
| slide images (`.tiff`) | PANDA release | PANDA image folder | thumbnails and patch crops of Figures 4, 9, 14 |
| fold checkpoints and feature caches | v20 notebooks; feature extraction in `01-data-preparation/` | `CLAM_TR_Checkpoints/`, `CLAM_TR_Features/` | calibration notebook (Figure 16) |
| thesis build (LaTeX sources, figure PDFs) | the thesis | `thesis/main/` under `CLAM_TR_PROJECT_DIR` | final checks of the plotting scripts; optional checks of `lora_planned_tests.py` (note A) |
| `r2c1_c2_results.json` (independent copy of the fold-order QWKs, computed from the Drive copies of the prediction files) and `lora_checks_out.txt` (log of an earlier check of the LoRA results) | the analysis of the companion journal paper (the first file is in the sibling repository CLAM-TR); an earlier verification run | `CLAM_TR_FOLD_QWK_COPY`, `CLAM_TR_LORA_CHECK_LOG` (section 2) | optional cross-checks inside `lora_planned_tests.py`, skipped when the files are absent |

The per-slide files carry PANDA image identifiers and ISUP labels, and the images and feature tensors fall under the
licences in the main README, so they are not copied here. (The LoRA Stage 2 validation predictions in
`05-experimental/lora_v1_results/` are the only per-slide files in the repository; they were released with its first
version.) `analysis/attention_fold0_statistics.py` could not be re-run
on 16 Nov 2026: one of its five inputs, the earlier rank-1 extraction file that it reads for comparison, is no longer
at its recorded location. Its saved outputs reproduce Figure 15 and match the values printed on p.64.

## 7. Earlier notebooks in this folder

`CLAM_TR_Thesis_Figures.ipynb` and `CLAM_TR_Thesis_Figures_NO_TITLES.ipynb` (April 2026) draw early versions of the
figures. None of their outputs is used in the final thesis. Their statistics cells take the fold lists of the
summary files in stored order, which pairs different folds (section 2, point 1); the fold-aligned analysis of
`analysis/` supersedes them.
