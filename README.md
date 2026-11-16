# CLAM-TR-thesis-code

Source code of the Master's thesis

**Attention-Enhanced Dual-Branch Aggregation for Whole Slide Image Classification in Prostate Cancer Grading**

- **Author.** Mohamed Fawzy Farahat Rateb Hassan
- **Supervisor.** Prof. Dr. Murat EKİNCİ
- **Institution.** Karadeniz Technical University, Graduate School of Natural and Applied Sciences, Department of
  Computer Engineering, Trabzon, Turkey
- **Degree.** Master's thesis (in English), 2026

The thesis PDF will be available through the KTU thesis catalogue after the defence.

---

## The model and the study

**CLAM-TR** stands for *CLAM with Top-K Selection and Retention*. It is an attention-enhanced dual-branch aggregator
for weakly supervised ISUP grade classification from prostate whole slide images. It pairs a CLAM-style
softmax-weighted-sum branch over the top-128 attended patches with an enhanced retention block applied to the same
patches, and the two branch outputs are fused by addition before classification. The retention block uses eight
softmax self-attention heads over the patch features (with optional 2D spatial decay over patch coordinates),
followed by a GroupNorm, a Swish-gated projection and a feed-forward enhancement module. The encoder stays frozen.

The thesis evaluates CLAM-TR in a 24-run factorial study (six aggregators × two UNI-family encoders × two
hyperparameter sets) on the 9,128-slide noise-cleaned PANDA cohort, with slide-level stratified five-fold
cross-validation; quadratic weighted kappa (QWK) is the primary metric. The six aggregators are ABMIL (the gated
attention baseline), DualBranch, RetNet, SpatialMult, SpatialAdd and SpatialMultRetNet; `RetNet` is the name of the
variant with self-attention heads and no spatial-decay mask. The encoders are UNI (ViT-L/16, 1024-d) and UNI2-h
(ViT-H/14, 1536-d).

## Contributions (final wording of the thesis)

1. **Attention-enhanced dual-branch aggregation.** On UNI, the best configuration (RetNet, best hyperparameters)
   reaches QWK 0.9543 ± 0.0051 against an ABMIL baseline at 0.9456 ± 0.0043, with a positive difference in all five
   folds (paired effect size d_z = +3.513; Nadeau–Bengio corrected p = 0.0064). Across the eleven pre-specified
   within-encoder comparisons, Benjamini–Hochberg gives q = 0.0699 with the Nadeau–Bengio correction (q = 0.0156 with
   the naive paired test). A three-seed replication run afterwards for the companion journal paper (15 seed–fold
   replicates) gave a gain of +0.0124 QWK, positive in 14 of the 15 replicates (corrected p = 0.013), but missed the
   pre-specified family-wise criterion (Holm-adjusted p = 0.0502). The gain is **consistent in direction but not
   confirmed family-wise**.
2. **Per-grade pattern.** The middle grades G2, G3 and G4 form the hardest tier: the minimum per-grade accuracy falls
   in G2 or G3 in twelve of the twelve best-hyperparameter configurations, while G0 is classified most accurately and
   G5 lies between 0.743 and 0.850. The parallel with inter-observer disagreement at these grades is descriptive.
3. **Encoder–aggregator interaction: tested and not supported.** The two encoders rank the aggregators differently,
   but the interaction test of a two-way repeated-measures ANOVA gives F(5, 20) = 1.310, p = 0.299, and the three-seed
   replication did not reproduce the rank reversal. The practical recommendation is to report the encoder together
   with any aggregator result.
4. **Cohort-scale observation.** At 600 development slides an earlier concatenation-fusion retention model did not
   differ significantly from a non-gated ABMIL (p ≈ 0.84), whereas at 9,128 slides RetNet differed from the gated
   ABMIL (paired t, p = 0.0014; Nadeau–Bengio corrected p = 0.0064). The two comparisons differ in model, baseline and
   learning rate as well as in cohort size, so they do not establish a sample-size threshold; the observation
   motivated the full-cohort design and the multi-seed replication.

**Scope.** All results are in-distribution results on PANDA only. The model is a research prototype for
methodological study; it has not been validated for clinical use.

**Exploratory LoRA branch.** Low-rank adaptation (LoRA) of the UNI encoder followed a pre-registered five-stage
protocol (Hassan & Ekinci, 2026, Zenodo, https://doi.org/10.5281/zenodo.20529009). Stages 0-4 were completed. The
best LoRA-UNI configuration reached 0.9529 ± 0.0026, below the best frozen-UNI configuration (0.9543): encoder
adaptation gave **no clear gain**, and the branch is not used as evidence for Contribution 3.

---

## Repository structure

```
CLAM-TR-thesis-code/
+-- 01-data-preparation/   subset CSVs, copy of the 9,128 slides, UNI feature extraction, UNI2-h feature download,
|                          LoRA Stage 3 feature re-extraction
+-- 02-development/        development iterations v1-v11
+-- 03-optimization/       optimisation and validation iterations v14-v19 (incl. the multi-scale variant that was
|                          considered and not adopted, thesis Section 4.8)
+-- 04-final-v20/          final model: six aggregator variants, 24-run factorial on 9,128 slides, UNI and UNI2-h
+-- 05-experimental/       exploratory LoRA fine-tuning (Stages 0-4) and its result files (lora_v1_results/)
+-- 06-thesis-figures/     statistics and figure scripts of the thesis
|   +-- analysis/          statistics scripts and the calibration notebook
|   +-- plots/             figure scripts
|   +-- tikz/              TikZ sources of the drawn figures
|   +-- data/              small derived input files (result summaries, cohort counts, calibration and LoRA results)
|   +-- TABLES_AND_FIGURES.md   every table and figure -> script -> data source -> status
|   +-- CLAM_TR_Thesis_Figures*.ipynb   early figure notebooks of April 2026 (superseded)
+-- 07-q1-sprint/          notebooks of the work for the companion journal paper (UNI features at 5x, TCGA-PRAD
|                          external validation, TCGA-PRAD benign slides) and two earlier analysis notebooks
|                          (retention-branch zero-out ablation, cohort-wide rank-1 attention extraction) whose
|                          results are not used in the thesis (Sections 3.6 and 4.2)
+-- 08-kaggle-panda-baseline/   reproduction notebooks of the PANDA 2020 1st-place solution and upstream copies
+-- REPRODUCIBILITY.md     software manifest, random-seed policy, run order and commands
+-- requirements.txt
+-- data_pointers.md       datasets, encoders and licences
+-- LICENSE                GPL-3.0-only
```

## How to reproduce

- **Run order, commands, software versions and seed policy:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
- **Which script produces which table or figure, from which data file:**
  [`06-thesis-figures/TABLES_AND_FIGURES.md`](06-thesis-figures/TABLES_AND_FIGURES.md).

The training notebooks run on Google Colab with a GPU. The statistics and figure scripts run on a CPU from the
repository root, for example

```
python -P 06-thesis-figures/analysis/fold_aligned_recompute.py
```

They read the per-fold prediction files of the training runs, which are not part of this repository because they
carry PANDA slide identifiers and labels (see `TABLES_AND_FIGURES.md`, sections 2 and 6). All paired statistics pair
the five folds by fold index. On 16 Nov 2026 the statistics scripts were re-run on the original result files and
the printed values of the data tables matched cell by cell; the figure scripts wrote PDFs identical to the figures in
the thesis. Two steps were not re-run (the Colab calibration notebook and one attention-statistics script); see
`REPRODUCIBILITY.md`, section 4.

## Data and licences

- **PANDA.** Obtained from the publicly available Kaggle Prostate Cancer Grade Assessment challenge archive under
  its official data-use agreement, https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data
  (Bulten et al., 2022). The development set is released under the Creative Commons Attribution-NonCommercial-
  ShareAlike 4.0 licence (CC BY-NC-SA 4.0).
- **9,128-slide noise-cleaned CV split.** Taken from the public release of the PANDA 1st-place solution
  (Team PND et al., 2020), https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution, licence CC-BY-NC 4.0.
- **UNI and UNI2-h encoders.** Public HuggingFace releases https://huggingface.co/MahmoodLab/UNI and
  https://huggingface.co/MahmoodLab/UNI2-h under the Mahmood Lab gated-research licence. The UNI features were
  extracted for this thesis from the PANDA patches with the released UNI ViT-L/16 weights.
- **UNI2-h features.** Downloaded pre-extracted from the public MahmoodLab/UNI2-h-features HuggingFace repository
  (Mahmood Lab, 2025), which publishes 1,536-dimensional features for the full PANDA cohort under the CC-BY-NC-ND 4.0
  licence. They were used under its non-commercial terms and are not redistributed.

No slide images, patch features, model weights or the PANDA label file are included here, and neither are the
per-slide prediction files of the factorial runs. The only per-slide files in the repository are the LoRA Stage 2
validation predictions in `05-experimental/lora_v1_results/`. Details: `data_pointers.md`.

## Hugging Face token

Hugging Face personal access tokens for the gated Mahmood Lab UNI and UNI2-h encoders are removed from all released
notebooks. Replace the placeholder `hf_YOUR_HF_TOKEN_HERE` with your own read-access token after accepting the
Mahmood Lab gated-research licence on the model cards.

## Companion journal paper

The scripts of the companion journal paper (under review), including the TCGA-PRAD external-validation analysis,
the decision-rule sensitivity sweep and the per-site forest plot, are released in the sibling repository
https://github.com/MohamedFarht/CLAM-TR under the same licence. The contributions of the thesis are in-distribution
results on PANDA; the external-validation results belong to the companion paper.

## Licence

GNU General Public License, version 3 only (GPL-3.0-only); see `LICENSE`. The licence follows that of CLAM
(https://github.com/mahmoodlab/CLAM), parts of which this code re-implements.

## How to cite

```
Hassan, M. F. F. R. (2026). Attention-Enhanced Dual-Branch Aggregation for Whole Slide Image Classification in
Prostate Cancer Grading [Master's thesis, Karadeniz Technical University].
```

```bibtex
@mastersthesis{hassan2026clamtr,
  author = {Hassan, Mohamed Fawzy Farahat Rateb},
  title  = {Attention-Enhanced Dual-Branch Aggregation for Whole Slide Image Classification in Prostate Cancer Grading},
  school = {Karadeniz Technical University, Graduate School of Natural and Applied Sciences},
  year   = {2026},
  type   = {Master's thesis}
}
```

## Contact

Mohamed Hassan: mohamed.fawzy98@hotmail.com
Murat EKİNCİ: ekinci@ktu.edu.tr
