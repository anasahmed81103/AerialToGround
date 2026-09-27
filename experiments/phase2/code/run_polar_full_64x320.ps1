# polar_full_64x320 — same as polar_full but native head 64x320 (not 32x160).
# Features: cache/dinov2s_reg_336 (unchanged). Labels: labels_64x320_train.npy.
$ErrorActionPreference = 'Stop'
python -u train_polar.py `
    --name polar_full_64x320 `
    --cache cache\dinov2s_reg_336 `
    --out_h 64 --out_w 320 `
    --batch_size 16 --num_workers 4 --epochs 8
python experiments\phase2\code\append_polar_full_64x320_results.py
