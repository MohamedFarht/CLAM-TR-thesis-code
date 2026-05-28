# 08-kaggle-panda-baseline

Kaggle PANDA 2020 1st-place solution code used as the published-benchmark
reference for this thesis (the "exceeds 0.9409" anchor for D23 in
DECISIONS.md).

## Layout

```
08-kaggle-panda-baseline/
├── README.md                                  (this file)
├── PANDA_EXACT_FINAL.ipynb                    our reproduction (full pipeline)
├── submitted_notebook_COMPLETE_FIX.ipynb      our reproduction (training)
├── panda_inference_FINAL_CORRECTED.ipynb      our reproduction (inference)
└── upstream/                                  upstream untouched
    ├── submitted_notebook.ipynb
    ├── train_famdata-kfolds.ipynb
    └── copy_subset_slides.ipynb               (originally in -master/input/)
```

The 3 root-level notebooks are our debugged reproductions of the 1st-place
solution. The `upstream/` folder holds the original notebooks shipped in
`Kaggle-PANDA-1st-place-solution-master/`. Provenance preserved so future
readers can diff our fixes against the upstream version.

Companion documentation for this baseline (debugging report, pipeline
architecture diagram, TIFF preprocessing notes) remains in
`Kaggle-PANDA-1st-place-solution/` at the project root.
