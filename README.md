# CLAM-TR-thesis: Source Code for the Master's Thesis

**Thesis title.** Deep Learning-Based Whole Slide Image Classification for Prostate Cancer Grading

**Author.** Mohamed HASSAN (Student #440638)
**Supervisor.** Prof. Dr. Murat EKINCI
**Institution.** Karadeniz Technical University, Graduate School of Natural and Applied Sciences (KTU FBE), Department of Computer Engineering, Trabzon, Turkey
**Program.** Master's (Yuksek Lisans), thesis in English
**Defence.** 2026

---

## Scope of this repository

This repository contains the **complete source code base** developed during the master's thesis, organised by experimental stage. The corresponding manuscript (PDF) is published through the KTU FBE thesis catalogue and through ULAKBIM TR Dizin.

For the **Q1 NCA paper** ("Encoder x Aggregator Co-Selection for Cross-Cohort Prostate Cancer Grading"), the figure- and table-generation scripts are released under a separate repository at [github.com/MohamedFarht/CLAM-TR](https://github.com/MohamedFarht/CLAM-TR). The two repositories overlap on the `07-q1-sprint/` notebooks (the thesis archive includes them here for completeness; the Q1 repository is the canonical citation for paper-level reproducibility).

---

## Repository structure

```
thesis-code/
+-- 01-data-preparation/    Feature extraction (UNI, UNI2-h) + PANDA noise-clean subset CSVs
+-- 02-development/         Architecture iterations v1, v2, v3, v4, v5, v7, v11
+-- 03-optimization/        Hyperparameter optimisation iterations v14-v19 incl. multi-scale 9128
+-- 04-final-v20/           Thesis final model (CLAM-TR v20) variants for both UNI and UNI2-h
+-- 05-experimental/        LoRA fine-tuning pre-registered protocol (Stages 0-2 complete)
+-- 06-thesis-figures/      Thesis figure-generation notebook (Ch.2-Ch.6 figures)
+-- 07-q1-sprint/           Q1 paper sprint (TCGA-PRAD external validation, benign extraction)
+-- 08-kaggle-panda-baseline/  PANDA 1st-place reproductions + upstream forks
```

---

## Key thesis findings, by chapter

| Chapter | Topic | Headline finding |
|---|---|---|
| Ch.1 Genel Bilgiler | Background + motivation | Prostate Gleason grading bottlenecks justify the dual-branch retention design |
| Ch.2 Yapilan Calismalar | Methodology | 6-aggregator x 2-encoder factorial on noise-cleaned PANDA (9,128 slides) |
| Ch.3 Bulgular | Results | RetNet-bestHP / UNI v1 attains $\kappa_w = 0.9543 \pm 0.0046$ on PANDA 5-fold CV |
| Ch.4 Tartisma | Discussion | Encoder x aggregator interact non-additively; rank reversal between UNI v1 and UNI2-h |
| Ch.5 Sonuclar | Conclusions | Aggregator and encoder must be co-selected; spatial-decay helps boundary classes |
| Ch.6 Oneriler | Future work | LoRA fine-tuning, multi-scale Mamba, cross-cohort external validation (Q1 paper) |

---

## Dependencies

All notebooks are designed to run on **Google Colab** (Tesla T4 or NVIDIA A100). Core dependencies:

```
torch >= 2.1.0
torchvision >= 0.16.0
timm >= 0.9.0
transformers >= 4.35.0
huggingface_hub >= 0.19.0
openslide-python >= 1.3.0
einops >= 0.7.0
scikit-learn >= 1.3.0
pandas, numpy, matplotlib, seaborn, tqdm
```

Each notebook prints its full `pip` install cell as Cell 1; standard Colab kernels already have most of these.

---

## Data sources

- **PANDA (training cohort).** Kaggle Prostate Cancer Grade Assessment Challenge: <https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data>. Licence: CC-BY-NC 4.0. We use the 9,128-slide noise-cleaned subset from the 1st-place team's release at <https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution>.
- **TCGA-PRAD (external test cohort, used in Q1 paper only).** NIH Genomic Data Commons: <https://portal.gdc.cancer.gov/projects/TCGA-PRAD>. Open access.
- **UNI / UNI2-h foundation encoders.** Mahmood Lab HuggingFace releases at <https://huggingface.co/MahmoodLab/UNI> and <https://huggingface.co/MahmoodLab/UNI2-h>. Licence: gated-research (must accept terms).

---

## Hugging Face token placeholder

Notebooks reference a placeholder `hf_YOUR_HF_TOKEN_HERE` in the cell that authenticates with the Mahmood Lab UNI encoder. Replace with your own token (read access is sufficient) after accepting the Mahmood Lab licence on the model card. The placeholder is intentionally present so that a fork of this repository can be re-authenticated without modification of the notebook structure.

---

## Citation

If you use this code, please cite the thesis:

```
Hassan, M. (2026). Deep Learning-Based Whole Slide Image Classification for
Prostate Cancer Grading. M.Sc. thesis, Karadeniz Technical University,
Graduate School of Natural and Applied Sciences, Trabzon, Turkey.
```

For the Q1 NCA paper companion:

```
Hassan, M., & Ekinci, M. (2026). Encoder x Aggregator Co-Selection for
Cross-Cohort Prostate Cancer Grading: External Validation on TCGA-PRAD.
Neural Computing and Applications (under review).
```

---

## Licence

GNU General Public License, version 3 only (GPL-3.0-only); see `LICENSE`. The licence follows that of CLAM (https://github.com/mahmoodlab/CLAM), parts of which this code re-implements.

---

## Contact

Mohamed HASSAN: mohamed.fawzy98@hotmail.com
Murat EKINCI: ekinci@ktu.edu.tr
