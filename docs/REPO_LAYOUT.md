# Repository layout

## Active (Phase 2 research + demo)

| Path | Role |
|------|------|
| `pipeline.py`, `app.py` | End-to-end inference + Gradio demo |
| `final_eval.py`, `bootstrap_ci.py` | Publication eval harness |
| `train_polar.py`, `polar_layout_model.py`, `cache_dino_features.py` | Stage 1 |
| `train_gan_pix2pix.py`, `eval_gan_pix2pix.py`, `spade_model.py`, `gan_unet_model.py` | Stage 2 |
| `aerial_warp.py`, `geo_calibrate.py`, `gan_*`, `semantic_metrics.py`, `phase2_*.py` | Shared |
| `tests/test_pipeline_parity.py` | Pipeline parity vs cache / final_eval |
| `experiments/phase2/` | Logs, metrics JSON, `MODELS.txt`, `README.md`, run scripts in `code/*.sh` |
| `experiments/final_eval/` | Final eval metrics (when present locally) |
| `experiments/RESEARCH_HANDOFF.md` | Paper / preprint handoff |

## Experiment archives (frozen snapshots)

| Path | Role |
|------|------|
| `experiments/baseline/` | Zhai-style CrossNet reimplementation + old labels |
| `experiments/phase1/` | SegFormer v2 labels + CrossNet v2 training snapshot |
| `experiments/phase2/code/` | **Shell/PowerShell launchers only** (Python runs from repo root) |

## Legacy

| Path | Role |
|------|------|
| `legacy/tensorflow/` | TF reference code |
| `legacy/phase1_v1/` | Old CrossNet+GAN inference/training entrypoints |

## Utilities

| Path | Role |
|------|------|
| `scripts/` | Smoke tests, sanity checks, dataset prep |
| `docs/reports/` | Course report, paper PDF, architecture notes (not required to run code) |

## Local-only (gitignored)

`outputs/`, `cache/`, `cvpr_subset/`, `experiments/**/runs/**/*.pt`, `test_pipeline_inputs/`
