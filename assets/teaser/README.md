# README teaser figures (Phase 2 best checkpoints)

| File | Source | Caption |
|------|--------|---------|
| `stage1_polar_layout.jpg` | `polar_full` test showcase @ **0.514 mIoU** | Aerial · GT pano · GT layout · **predicted layout** |
| `stage2_spade_aerial_layout_real_fake.jpg` | `G_step0032000` (**FID 116.2** oracle) | GT layout · real pano · **SPADE+aerial fake** |
| `e2e_aerial_gen_gt_val500.png` | `pipeline.py` Config **E** | Aerial · **E2E generated** · ground truth |
| `e2e_full_montage_val500.png` | same run | aerial · layout · warp · pano |

Regenerate Stage 2 strip: `python experiments/phase2/code/make_gan_showcase_dumps.py --preset spade_aerial_v2 --n_images 12` (pick best frame).
