# Zenodo bundle — release v1.0.0

Upload these files as a single `.zip` (or tar) attached to **GitHub release v1.0.0**; Zenodo mirrors the release when integration is enabled.

## Required (reproduce `pipeline.py` and Config **E**)

| Path in repo | Approx. size | SHA note |
|--------------|--------------|----------|
| `experiments/phase2/runs/polar_full/best.pt` | ~6 MB | Stage 1 |
| `experiments/phase2/runs/spade_aerial_v2/ckpts/G_step0032000.pt` | ~1.0 GB | Stage 2 generator |

**Uncompressed total (minimal): ~1.06 GB.** With zip compression, often ~0.35–0.45 GB depending on tool.

Extract preserving directory names so defaults in `pipeline.py` and `MODELS.txt` resolve.

## Optional (eval ladder A–E / ablations)

| Path | Role |
|------|------|
| `experiments/phase2/runs/spade_v2/ckpts/G_step0033300.pt` | Config B/C family (~1.0 GB) |
| `experiments/phase2/runs/gan_v2_random_d/ckpts/G_step0231066.pt` | Config A Pix2Pix (~623 MB) |

Discriminators (`D_*.pt`) are not required for inference.

## Not included

- CVUSA tiles or panos (request from [CVUSA](https://mvrl.cse.wustl.edu/datasets/cvusa/))
- DINOv2 weights (downloaded from Hugging Face on first run)
- Full DINO feature cache (`cache/dinov2s_reg_336/`)

## Suggested zip layout

```
Aerial2Ground-checkpoints-v1.0.0/
  experiments/phase2/runs/polar_full/best.pt
  experiments/phase2/runs/spade_aerial_v2/ckpts/G_step0032000.pt
  README.txt   # copy MODELS.txt + one-line install path hint
```

After Zenodo assigns a DOI, add it to `CITATION.cff` under `identifiers:` and tag metadata on GitHub release.
