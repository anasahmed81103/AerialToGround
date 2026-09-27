# Sequential GPU queue: fair full-test evals of old models, then Phase 2 ablations.
$ErrorActionPreference = 'Continue'

python -u eval_crossnet.py --ckpt experiments\baseline\checkpoints\crossnet_step0088830.pt `
    --val_csv cvpr_val_v2.csv --no_amp --save_n 4 `
    --dump_dir experiments\baseline\results\eval_v2_labels_full8884

python -u eval_crossnet.py --ckpt experiments\phase1\checkpoints\crossnet_best.pt `
    --val_csv cvpr_val_v2.csv --no_amp --save_n 4 `
    --dump_dir experiments\phase1\results\eval_v2_labels_full8884

foreach ($abl in @(
    @('polar_prior_none', '--prior', 'none'),
    @('polar_prior_fixed', '--prior', 'fixed'),
    @('polar_no_offsets', '--no_offsets')
)) {
    $name = $abl[0]
    $extra = $abl[1..($abl.Length - 1)]
    python -u train_polar.py --name $name --batch_size 16 --num_workers 4 `
        --out_h 32 --out_w 160 --epochs 8 @extra
}
Write-Output '[queue] all done'
