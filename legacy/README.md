# Legacy code (not used by the Phase 2 pipeline)

| Folder | Contents |
|--------|----------|
| `tensorflow/` | Original TensorFlow CrossNet reference (`*_tf_legacy.py`) |
| `phase1_v1/` | Phase 1 **CrossNet + v1 Pix2Pix** inference and training (8×40 grid, root quickstart) |

**Current inference:** use repo-root `pipeline.py` (DINO → PolarLayoutNet → SPADE+aerial).

Frozen copies of baseline / Phase 1 trainers also live under `experiments/baseline/code/` and `experiments/phase1/code/`.
