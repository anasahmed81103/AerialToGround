# Resume polar_full_64x320 from epoch 2 (epoch 1 already on disk). ~35 min/epoch on P1000 @ bs12.
$ErrorActionPreference = 'Stop'
python -u train_polar.py `
    --name polar_full_64x320 `
    --cache cache\dinov2s_reg_336 `
    --out_h 64 --out_w 320 `
    --batch_size 12 --num_workers 2 --epochs 8 `
    --resume experiments\phase2\runs\polar_full_64x320 `
    --save_n 4
python experiments\phase2\code\append_polar_full_64x320_results.py
