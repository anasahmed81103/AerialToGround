# Paper TODO

Every `\todo{}` in the PDF must have a line here. Remove the line when the `\todo{}` is resolved.

## Build
- [ ] No LaTeX distribution on this machine (no `latexmk`/`pdflatex`). Install MiKTeX or TeX Live and run
      `latexmk -pdf main.tex` inside `paper/`; fix any errors/overfull boxes (tables were never compiled).

## Front matter
- [ ] `main.tex`: author names, affiliations, emails.
- [ ] `sections/acknowledgements.tex`: funding / compute acknowledgements.
- [ ] `sections/acknowledgements.tex`: AI-assistance disclosure text.

## References
- [ ] `references.bib` is empty: no `.bib` file existed anywhere in the repo. Export entries
      (publisher / arXiv / DBLP) for every work the paper will cite; do not write them from memory.

## Compute (tables/compute.tex, numbers.tex `\GpuNameSOne*`)
- [ ] GPU type for Stage 1 runs `polar_prior_none`, `polar_prior_fixed`, `polar_no_offsets`,
      `polar_dense_mapping`, `polar_crop224`, `polar_vgg16`, `polar_full_classweighted`.
      `results.json` does not record it. Only `polar_full` (Quadro P1000 4GB) and
      `polar_full_64x320` (RTX 4090) are documented in `results_table.json`.
- [ ] GPU type for `final_eval` (metrics JSONs do not record it; RESEARCH_HANDOFF.md says RTX 4090 — verify).
- [ ] `spade_aerial_v2`: epochs 1–5 have no `epoch done` timing lines in `logs/spade_aerial_v2.txt`
      (first segment crashed, run resumed from step 12000). GPU-hours for those epochs are missing.
- [ ] Final-eval wall-clock excludes clean-FID/KID time (`elapsed_sec` is measured before FID).
      RESEARCH_HANDOFF.md says "~6 h" for the whole pod session, but no log records it.
- [ ] Time for DINOv2 feature caching and SegFormer relabelling GPU (relabel = 132 min, GPU not recorded).

## Data provenance to confirm
- [ ] `experiments/final_eval/` contains only a 64-image smoke run of Config E. Full results are in
      `final_eval_results/experiments/final_eval/`. make_numbers.py picks the full set automatically;
      consider copying it into `experiments/final_eval/` so the README path is correct.
- [ ] Stage 1 numbers are read from `experiments/phase2/runs/*/results.json` (full precision, includes
      params) and cross-checked against `results_table.json` (they agree).
