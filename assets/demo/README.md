# README demo figures

Tracked E2E runs from `pipeline.py` on CVUSA val rows (same checkpoints as Config **E**).

| File | Val CSV row | Panels |
|------|-------------|--------|
| `montage_val_00500.png` | 500 | aerial · layout · warped aerial · generated pano |
| `compare_val_00500.png` | 500 | aerial · generated · ground truth |

Regenerate:

```bash
python pipeline.py --aerial test_pipeline_inputs/aerial_00500.jpg --out outputs/dataset_pipeline_test/idx_00500
```
