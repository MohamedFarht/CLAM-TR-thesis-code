# 08-kaggle-panda-baseline

Notebooks of the PANDA 2020 1st-place solution (Team PND et al., 2020), whose public release provides the 9,128-slide
noise-cleaned five-fold split used in the thesis. The Kaggle PANDA 2020 winning private-leaderboard QWK of 0.9409 is
quoted in the thesis as context only: it was obtained on a held-out, expert-labelled test set under a different
protocol and is not compared with the thesis results.

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

The three root-level notebooks are our debugged reproductions of the 1st-place solution. The `upstream/` folder holds
the original notebooks shipped in `Kaggle-PANDA-1st-place-solution-master/`, so that the fixes can be compared with
the upstream version. The upstream solution is available at
https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution (licence CC-BY-NC 4.0); its extra Python packages are
listed at the end of `requirements.txt`. Notes written while debugging the reproduction are not part of this
repository.
