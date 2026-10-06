# diag3: random-init D + g_steps=3 + FP32 (local / Windows). Use bs=4 on 4GB GPUs.
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
Set-Location $Root
New-Item -ItemType Directory -Force -Path logs,
  "experiments/phase2/runs/gan_v2_diag3/ckpts",
  "experiments/phase2/runs/gan_v2_diag3/dump" | Out-Null

$bs = if ($env:GAN_DIAG3_BS) { $env:GAN_DIAG3_BS } else { 4 }
$env:PYTHONIOENCODING = "utf-8"

python -u train_gan_pix2pix.py `
  --train_csv cvpr_train_v2.csv `
  --resume_g outputs/ckpts_gan/G_step0197766.pt `
  --random_init_d `
  --g_steps 3 `
  --no_amp `
  --epochs 2 `
  --batch_size $bs `
  --num_workers 2 `
  --d_diag_max_steps 4500 `
  --ckpt_dir experiments/phase2/runs/gan_v2_diag3/ckpts `
  --dump_dir experiments/phase2/runs/gan_v2_diag3/dump `
  2>&1 | Tee-Object -FilePath logs/gan_v2_diag3.txt
