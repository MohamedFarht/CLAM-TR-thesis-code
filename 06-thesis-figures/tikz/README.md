# TikZ sources of the drawn figures

Each file is a complete standalone LaTeX document (`standalone` class, Times via `newtxtext`/`newtxmath`, TikZ,
`amsmath`; one file also uses `enumitem`). Compile a figure with

```
pdflatex fig_2_1_clam_tr_architecture.tex
```

The result is the PDF that the thesis includes with `\includegraphics`. A recent TeX Live or MiKTeX installation with
the packages above is enough; no bibliography run is needed.

| File | Thesis figure | p. |
|---|---|---|
| `fig_1_Y_inverted_triangle.tex` | Figure 1, inverted-triangle problem statement | 1 |
| `fig_3_cohort_flow.tex` | Figure 3, cohort flow | 20 |
| `fig_2_5_hybrid_precedent.tex` | Figure 6, hybrid-head design precedent | 27 |
| `fig_2_1_clam_tr_architecture.tex` | Figure 7, CLAM-TR architecture overview | 29 |
| `fig_2_2_preprocessing_flow.tex` | Figure 8, preprocessing and feature-caching pipeline | 30 |
| `fig_2_4_retention_branch_detail.tex` | Figure 10, retention-branch sub-components | 36 |
| `fig_2_6_lora_v1_pipeline.tex` | Figure 11, exploratory LoRA pipeline, Stages 0-4 | 46 |
| `fig_5_1_contributions_overview.tex` | Figure 19, synthesis of the contributions | 94 |
| `fig_6_2_future_roadmap.tex` | Figure 20, future-work roadmap | 99 |

The file names follow the names of the figure files in the thesis build, which keep the chapter numbering of an
earlier draft. Figure 3 is drawn inline in the thesis source (Chapter 2); `fig_3_cohort_flow.tex` holds the same
`tikzpicture` with a preamble that loads the TikZ libraries of the thesis and prints its two citations in the
author-year form of the thesis. The comment block at the top of each file names the thesis figure, lists the facts
the drawing was checked against (thesis text, model code, cited papers) and gives the font sizes used.

Check of 16 Nov 2026 (MiKTeX pdflatex): all nine files compile without errors or overfull boxes; the eight figure
files give PDFs with the same page size and the same words as the figure files of the thesis build, and the words of
`fig_3_cohort_flow.pdf` equal the text of the printed Figure 3.
