# Phase 2, ablation round 2: which of {polar geometry, full-tile context, DINOv2} carries the gain.
# Each run changes exactly one thing vs polar_full. Feature caches are deleted after use (regenerable).
$ErrorActionPreference = 'Continue'
$common = @('--batch_size', '16', '--num_workers', '4', '--out_h', '32', '--out_w', '160', '--epochs', '8')

# A) learned dense mapping instead of polar geometry (same DINOv2 full-tile features)
python -u train_polar.py --name polar_dense_mapping --mapping dense @common

# B) only the 224px center crop CrossNet saw (native resolution, 16x16 tokens)
python -u cache_dino_features.py --tag dinov2s_reg_crop224 --crop 224 --size 224
python -u train_polar.py --name polar_crop224 --cache cache\dinov2s_reg_crop224 @common
Remove-Item -Recurse -Force cache\dinov2s_reg_crop224

# C) frozen ImageNet VGG-16 relu5_3 instead of DINOv2 (same full tile at 336px, 21x21)
python -u cache_dino_features.py --tag vgg16_336 --model_id vgg16 --size 336
python -u train_polar.py --name polar_vgg16 --cache cache\vgg16_336 @common
Remove-Item -Recurse -Force cache\vgg16_336

Write-Output '[queue] all done'
