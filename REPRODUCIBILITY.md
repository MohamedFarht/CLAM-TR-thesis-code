# Reproducibility

Software-version manifest, random-seed policy and run order of the code for the thesis *Attention-Enhanced Dual-Branch
Aggregation for Whole Slide Image Classification in Prostate Cancer Grading*. Page numbers refer to the printed
thesis. The table-by-table and figure-by-figure map is `06-thesis-figures/TABLES_AND_FIGURES.md`.

## 1. Software and hardware

**Factorial study (01-04).** The full 9,128-slide factorial study was run on Google Colab Pro+ with NVIDIA Tesla T4
GPUs (15.6 GB device memory) for both encoders, using PyTorch 2.x, timm, HuggingFace Hub (for the gated
MahmoodLab/UNI and MahmoodLab/UNI2-h releases), OpenSlide, h5py, scikit-learn, scipy and seaborn (thesis p.44). The
saved outputs of the v20 notebooks show a Tesla T4. The setup cell of the v20 Colab notebooks installs, without
version pins:

- system packages (apt-get): `openslide-tools`, `libgl1-mesa-glx`;
- Python packages (pip): `openslide-python`, `timm`, `huggingface_hub`, `h5py`, `opencv-python-headless`.

All other packages (`torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`,
`tqdm`, `Pillow`) were those of the Colab image at run time. The notebooks do not print the package versions of the
runtime, so the exact versions of the training runs are not recorded.

Training settings that matter for repetition (thesis p.40-42 and Table 8): learning rate 2 × 10^-4 for both encoders
and both hyperparameter sets, selected by a single-fold ABMIL_bestHP sweep; CosineAnnealingLR with T_max = 50 and
minimum learning rate 0, stepped after every epoch; gradient clipping at norm 1.0; mixed precision with
`torch.amp.autocast('cuda')` and `torch.amp.GradScaler`; early stopping on the held-out fold QWK with patience 10 and
threshold 0.001, at most 50 epochs; batch size one slide; class weights from the class distribution of Table 4. UNI
features were cached as `.pt` files (fp32 features and coordinates on disk, features packed to fp16 on the GPU during
training) and UNI2-h features as `.h5` files (p.44). Checkpointing is atomic at the fold level, so a run can resume
without re-training a completed fold (p.44).

**Exploratory LoRA branch (05).** NVIDIA A100 80 GB with about 179 GB system RAM (p.44); it is not part of the primary
factorial. The notebook installs `torchao>=0.16.0` and uses `peft` for the LoRA adapters.

**Analysis and figure scripts (06-thesis-figures).** CPU only. The verification runs of 16 Nov 2026 used Python
3.13.3 with numpy 2.2.6, scipy 1.15.3, pandas 3.0.1, scikit-learn 1.8.0, statsmodels 0.15.0, matplotlib 3.10.8,
seaborn 0.13.2, PyMuPDF 1.27.2 and Pillow 11.2.1; the TikZ figures were compiled with pdflatex (MiKTeX).

`requirements.txt` lists the packages. Only `torchao` is pinned in a notebook; the other entries carry no exact
version because none was recorded.

## 2. Random-seed policy

- Thesis (p.42): "SEED=42 is fixed with torch.backends.cudnn.deterministic=True across all code paths; it sets model
  initialisation (the folds come from the released split file), but under mixed precision a re-run is not bit-wise
  identical."
- In the code: the seed cell of the v20 notebooks sets `SEED = 42` and calls `random.seed(SEED)`,
  `np.random.seed(SEED)`, `torch.manual_seed(SEED)` and `torch.cuda.manual_seed_all(SEED)`, and sets
  `torch.backends.cudnn.deterministic = True` (cell 4). These generators are seeded once per session, before the
  experiment loop, not again per fold or per experiment. Two feature-check cells (cells 8 and 11) re-seed Python's
  `random` module with 42 before they draw their sample of feature files.
- The five folds are not drawn with this seed. They are the released five-fold split of the 9,128-slide
  noise-cleaned cohort of the PANDA 1st-place solution (Team PND et al., 2020), stratified on Gleason score and
  provider, with each group of duplicate images in one fold (thesis Figure 3; `data_pointers.md`).
- The percentile bootstrap intervals use 10,000 resamples (p.43) with `np.random.RandomState(42)`, created afresh for
  each comparison (`06-thesis-figures/analysis/`), so the intervals are the same on every run.
- The LoRA notebook uses the same seed but sets `torch.backends.cudnn.deterministic = False` and
  `torch.backends.cudnn.benchmark = True` (marked in the notebook as a speed setting).
- What cannot be repeated exactly: under mixed precision a re-run is not bit-wise identical (p.42). A later
  retraining of the UNI2-h ABMIL_bestHP fold 0 with the same seed gave QWK 0.9476 instead of the original 0.9500
  (p.49); the thesis numbers use the original run.
- The UNI bestHP runs trained their five folds in parallel (ThreadPoolExecutor with up to five CUDA streams on one
  T4), while most UNI2-h runs trained their folds one after another (p.41).

## 3. Run order and commands

Data and licences: `data_pointers.md`. Hugging Face access: accept the Mahmood Lab gated-research licence for
MahmoodLab/UNI and MahmoodLab/UNI2-h and put your own read-access token where the notebooks show the placeholder
`hf_YOUR_HF_TOKEN_HERE`.

**Stage 1, data preparation (`01-data-preparation/`).**
1. Download the PANDA training images and `train.csv` from Kaggle, and the fold files from the public release of the
   1st-place solution (`train-5kfold_remove_noisy_by_0622_rad_13_08_ka_15_10.csv` holds the 9,128 slides).
2. `copy_subset_slides.ipynb` copies the 9,128 slides of that file out of the full image folder.
3. `Create_Subset_600_CSV.ipynb` builds the 600-slide development subset used by the notebooks in `02-development/`
   and `03-optimization/`.

**Stage 2, features (`01-data-preparation/`).**
1. `UNI_Feature_Extraction.ipynb` extracts the 1024-d UNI (ViT-L/16) features (`.pt`, one file per slide).
2. `download_features.ipynb` downloads the pre-extracted 1536-d UNI2-h features of the MahmoodLab/UNI2-h-features
   repository (`.h5`); they are used under their non-commercial terms and are not redistributed.

**Stage 3, v20 factorial training per encoder (`04-final-v20/`).** Open the notebook in Colab with a GPU, set the
token, run all cells in order. Each notebook writes `v20_final_summary.json`, `predictions/`, `checkpoints/`,
`best_lr.json` and `statistical_tests.json` to `CLAM_TR_Results/v20_fullscale/<encoder>/` on the mounted Drive.
- UNI: `CLAM_TR_v20_UNI_Colab.ipynb`, with `CLAM_TR_v20_UNI_Colab_OPTIMIZED.ipynb` writing to the same folder.
- UNI2-h: `CLAM_TR_v20_UNI2h_Colab.ipynb` (eleven configurations) and `CLAM_TR_v20_SpatialAdd_bestHP.ipynb`
  (SpatialAdd_bestHP).
- `CLAM_TR_v20_FullScale_Colab.ipynb` runs both encoders one after the other; `CLAM_TR_v20_FullScale_Local.ipynb` is
  the same notebook for a local machine.
- The paired tests in `statistical_tests.json` and in the analysis cells of these notebooks pair the per-fold values
  by their position in a list stored in completion order. The thesis replaces them by the fold-aligned analysis of
  Stage 4 (p.37, p.54).
- The thesis also notes (p.39) that the GPU-packing path of the headline notebook stores coordinates in FP16, which
  overflows for the 15 slides with coordinates above 65,504 px; the reported predictions for these slides are
  finite and agree with an FP32 forward pass, so the reported results are unaffected. Keep coordinates in FP32 for
  new runs.

**Stage 3b, after training (Colab).**
- Calibration (Figure 16, Section 3.7): `06-thesis-figures/analysis/CLAM_TR_v20_calibration.ipynb`, a copy of the
  v20 notebook cells with an appended cell 59. Run the setup cells, then cell 59: it detects the cached encoder,
  reloads the best-epoch checkpoints of the encoder-best configuration, requires at least 99% agreement with the
  saved predictions, and writes `calibration_results.json` (shipped as
  `06-thesis-figures/data/calibration_results.json`).
- Attention dumps (Figures 4, 9, 14, 15): written by an extra cell run in the Colab copy of
  `CLAM_TR_v20_FullScale_Colab.ipynb` after training; that cell is not part of the released notebook and the dumps
  are not redistributed (`06-thesis-figures/TABLES_AND_FIGURES.md`, section 6).

**Stage 4, statistics (`06-thesis-figures/analysis/`, CPU).** Set `CLAM_TR_V20_DIR` to the `v20_fullscale` results
folder (it must contain the per-slide `predictions/` folders), and `CLAM_TR_PROJECT_DIR` where a script needs the
PANDA label file. Then, from the repository root:

```
python -P 06-thesis-figures/analysis/fold_aligned_recompute.py                  # Tables 11-13, ANOVA, Appendix Table 3
python -P 06-thesis-figures/analysis/verify_chapter3_numbers.py                 # Tables 9-19
python -P 06-thesis-figures/analysis/recompute_summary_and_appendix_numbers.py  # Appendix Tables 2-3, Summary
python -P 06-thesis-figures/analysis/sd_ddof1_and_t_intervals.py                # Table 9 SDs, Table 10
python -P 06-thesis-figures/analysis/fold_aligned_contrasts_figs12_17_18.py     # data of Figures 12, 17, 18
python -P 06-thesis-figures/analysis/lora_planned_tests.py                      # Appendix Table 1
python -I 06-thesis-figures/analysis/cohort_flow_counts.py                      # Tables 4-5, Figures 3 and 5
```

`lora_planned_tests.py` also compares its values with the thesis LaTeX sources and with two further files when they
are available; without them, as in this repository, it skips those checks with a printed note
(`06-thesis-figures/TABLES_AND_FIGURES.md`, note A).

**Stage 5, figures (`06-thesis-figures/plots/` and `tikz/`).** The plotting scripts need the inputs listed in
`TABLES_AND_FIGURES.md` and the thesis build for their final checks:

```
python -P 06-thesis-figures/plots/make_figs_12_17_18.py
python -P 06-thesis-figures/plots/make_fig13_epoch_distribution.py
python -P 06-thesis-figures/plots/make_fig15_failure_mode_rank1.py
python -P 06-thesis-figures/plots/make_fig16_calibration_reliability.py
python -P 06-thesis-figures/plots/lineage_timeline_base.py
python -P 06-thesis-figures/plots/make_fig2_lineage_timeline.py
python -P 06-thesis-figures/plots/make_fig5_cv_split_schematic.py
python -P 06-thesis-figures/plots/make_appendix_figs_1_2_3.py
python -P 06-thesis-figures/plots/make_fig4_isup_samples.py
python -P 06-thesis-figures/plots/make_fig9_topk_selection.py
python -P 06-thesis-figures/plots/make_fig14_top_patches.py
cd 06-thesis-figures/tikz && pdflatex <file>.tex
```

**Exploratory LoRA branch.** `05-experimental/CLAM_TR_LoRA_v1_FIXED_v8_SHARDED.ipynb` (A100) runs Stages 0-4;
`01-data-preparation/LoRA_Feature_Extraction_FINAL_v4.ipynb` is the held-out Stage 3 feature re-extraction. The
small result files are in `05-experimental/lora_v1_results/`. Then run `analysis/lora_planned_tests.py` and
`plots/make_appendix_figs_1_2_3.py`.

## 4. What was checked on 16 Nov 2026

Every statistics script above was run from the repository working tree on the original result files. The printed
values of Tables 4, 7, 9-19 and Appendix Tables 1-3 were compared cell by cell with the thesis text and matched (one
known difference of the short script `full_factorial_24_runs.py` is explained in `TABLES_AND_FIGURES.md`). Every
plotting script produced a PDF byte-identical to the file in the thesis build, and every TikZ source compiled to a
figure with the same page size and words. Two steps were not re-run: the calibration notebook (Colab GPU) and the
attention statistics script (one input file is no longer at its recorded location); their saved outputs match the
numbers printed on p.64 and p.67.
