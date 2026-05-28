# Data Pointers

## PANDA (primary training cohort)

- **Source.** Kaggle Prostate cANcer graDe Assessment (PANDA) Challenge.
- **URL.** <https://www.kaggle.com/c/prostate-cancer-grade-assessment>
- **Licence.** CC-BY-NC 4.0.
- **Original size.** 10,616 slides (Karolinska University Hospital + Radboud University Medical Center).
- **Noise-cleaning step.** Filter out the slides flagged as noisy in the dataset providers' supplementary list (Bulten et al., *Nature Medicine* 2022). Resulting cohort: **9,128 slides** spanning ISUP grades 0--5.
- **5-fold split.** Stratified by ISUP grade, fixed random seed (42). Used by the 1st-place team's released split at <https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution>.

## UNI v1 encoder

- **Source.** Mahmood Lab (Chen et al., *Nature Medicine* 2024).
- **URL.** <https://huggingface.co/MahmoodLab/UNI>
- **Licence.** Gated repository; researchers must accept terms before download.
- **Use in thesis.** Frozen feature extractor; 1,024-dimensional ViT-L/16 embeddings per 256x256-pixel patch at 20x effective magnification.

## UNI2-h encoder

- **Source.** Mahmood Lab.
- **URL.** <https://huggingface.co/MahmoodLab/UNI2-h>
- **Licence.** Gated repository.
- **Use in thesis.** Frozen feature extractor; 1,536-dimensional ViT-H/14 embeddings. We use the pre-extracted feature tensors from the public `MahmoodLab/UNI2-h-features` HuggingFace repository (saves Colab time vs. re-extraction).

## TCGA-PRAD (external test cohort -- used in Q1 paper only; thesis is PANDA-only)

- **Source.** NIH Genomic Data Commons.
- **URL.** <https://portal.gdc.cancer.gov/projects/TCGA-PRAD>
- **Licence.** NIH open-access (no controlled-access components used).
- **Cancer-positive cohort.** 426 Diagnostic Slides (sample type code 01, Primary Tumor; DX-prefixed).
- **Benign cohort.** 118 Tissue Slides (sample type code 11, Solid Tissue Normal; TS-prefixed).

The TCGA-PRAD GDC REST API queries used to retrieve both subcohorts are embedded in the Q1 paper Supplementary Section S7 (see [github.com/MohamedFarht/CLAM-TR](https://github.com/MohamedFarht/CLAM-TR) for the Q1 paper code release).

## Reproducibility convention

All preprocessing, feature extraction, training, and evaluation derive randomness from a single seed (42). The release Stage 4 LoRA fine-tuning protocol uses seeds 1 through 20 for the multi-seed extension (deferred to post-defence per the thesis future-work programme).
