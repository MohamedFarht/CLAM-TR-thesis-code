# Data pointers

Where the data, encoders and features used by the thesis come from, and under which licences. None of these files
is redistributed in this repository. Page numbers refer to the printed thesis.

## PANDA (the cohort of the thesis)

- **Source.** Kaggle Prostate Cancer Grade Assessment (PANDA) challenge archive, under its official data-use
  agreement: <https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data> (Bulten et al., 2022).
- **Licence.** The development set is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  licence (CC BY-NC-SA 4.0) (p.47).
- **Raw release.** 10,616 slides: Karolinska 5,456 + Radboud 5,160 (thesis Figure 3).
- **Noise-cleaned subset.** 9,128 slides (Karolinska 5,121 + Radboud 4,007), made by model-based label denoising in
  the PANDA 1st-place solution: 1,488 slides on which the Team PND model disagreed with the label were removed
  (Radboud 1,153, Karolinska 335) (thesis Figure 3 and Limitation L1). The subset favours models that agree with the
  cleaning model, so in-distribution QWK values are likely higher than on the full release (Limitation L1).
- **Five-fold split.** Taken from the public release of the 1st-place solution (Team PND et al., 2020),
  <https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution>, licence CC-BY-NC 4.0 (p.47); file
  `train-5kfold_remove_noisy_by_0622_rad_13_08_ka_15_10.csv`. The split is stratified on Gleason score and provider,
  with slide-level isolation and each group of duplicate images in one fold; fold sizes 1,803 / 1,787 / 1,816 /
  1,933 / 1,789 (thesis Figure 3 and Section 2.1.3). The same five folds are used for every configuration.

## UNI encoder (ViT-L/16)

- **Source.** Mahmood Lab (Chen et al., 2024), <https://huggingface.co/MahmoodLab/UNI>.
- **Licence.** Mahmood Lab gated-research licence; accept the terms on the model card before downloading (p.47).
- **Use in the thesis.** Frozen feature extractor. The UNI features were extracted for this thesis from the PANDA
  slides with the released weights (`01-data-preparation/UNI_Feature_Extraction.ipynb`): non-overlapping
  256 × 256-pixel patches at 20× (level 0), kept if at least 50% tissue, resized to 224 × 224 pixels, 1,024-d feature
  per patch, one `.pt` file per slide with the features and the level-0 coordinates; median 457 patches per slide
  (thesis Figure 8 and Section 2.3.2).

## UNI2-h encoder (ViT-H/14)

- **Source.** Mahmood Lab, <https://huggingface.co/MahmoodLab/UNI2-h>; licence: Mahmood Lab gated-research licence
  (p.47).
- **Features.** The UNI2-h features were downloaded pre-extracted from the public MahmoodLab/UNI2-h-features
  HuggingFace repository (Mahmood Lab, 2025; `01-data-preparation/download_features.ipynb`), which publishes
  1,536-dimensional features for the full PANDA cohort under the CC-BY-NC-ND 4.0 licence. They were used under its
  non-commercial terms and are not redistributed (p.47). One `.h5` file per slide; median 538 patches per slide
  (Section 2.3.2). The two encoders therefore saw different patch sets from different extraction pipelines
  (Limitation L2).

## TCGA-PRAD (companion journal paper only)

The thesis results are in-distribution PANDA results. The zero-shot TCGA-PRAD evaluation (NIH Genomic Data
Commons, <https://portal.gdc.cancer.gov/projects/TCGA-PRAD>) belongs to the companion journal paper (under review);
its cohort definition and scripts are in the sibling repository <https://github.com/MohamedFarht/CLAM-TR>. The
notebooks in `07-q1-sprint/` were written for that work.

## Random seed

The v20 factorial notebooks and the LoRA notebook fix `SEED = 42`; under mixed precision a re-run is not bit-wise
identical (p.42). The folds
come from the released split file above, not from the seed. Details: `REPRODUCIBILITY.md`.
