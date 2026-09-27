"""
eval_crossnet.py  —  Score a CrossNet checkpoint on the validation split.

Usage:
    python eval_crossnet.py
    python eval_crossnet.py --ckpt outputs/ckpts_pt/crossnet_best.pt
    python eval_crossnet.py --ckpt outputs/ckpts_pt/crossnet_step0088830.pt --max_samples 256
"""

import argparse
import csv
import glob
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from crossnet_model import CrossNet
from crossnet_dataset import CVPRDataset
from semantic_metrics import (
    CLASS_NAMES,
    evaluate_crossnet,
    format_metrics,
    metrics_row,
    save_eval_montage,
)


ROOT = os.path.dirname(os.path.abspath(__file__))


def _latest_ckpt(ckpt_dir: str) -> str:
    best = os.path.join(ckpt_dir, 'crossnet_best.pt')
    if os.path.isfile(best):
        return best
    hits = sorted(glob.glob(os.path.join(ckpt_dir, 'crossnet_step*.pt')))
    return hits[-1] if hits else ''


def get_args():
    p = argparse.ArgumentParser(description='Evaluate CrossNet mIoU on the val split')
    p.add_argument('--ckpt', default='',
                   help='Checkpoint .pt. Default: crossnet_best.pt, else latest step ckpt')
    p.add_argument('--ckpt_dir', default=os.path.join(ROOT, 'outputs', 'ckpts_pt'))
    p.add_argument('--val_csv', default='cvpr_val.csv')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--num_classes', type=int, default=4)
    p.add_argument('--max_samples', type=int, default=0,
                   help='Evaluate only the first N val images (0 = all)')
    p.add_argument('--save_n', type=int, default=8,
                   help='Save this many colorized qualitative montages (0 = none)')
    p.add_argument('--dump_dir', default=os.path.join(ROOT, 'outputs', 'eval_pt'))
    p.add_argument('--no_conditioned', action='store_true')
    p.add_argument('--no_amp', action='store_true')
    return p.parse_args()


def main():
    args = get_args()
    ckpt = args.ckpt or _latest_ckpt(args.ckpt_dir)
    if not ckpt or not os.path.isfile(ckpt):
        raise SystemExit(
            'No CrossNet checkpoint found. Pass --ckpt path/to/model.pt '
            f'(looked in {args.ckpt_dir}).'
        )
    if not os.path.isfile(args.val_csv):
        raise SystemExit(f'Val CSV not found: {args.val_csv}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'
    print(f'[*] Device : {device}', flush=True)
    print(f'[*] AMP    : {"ON" if use_amp else "OFF"}', flush=True)
    print(f'[*] Ckpt   : {ckpt}', flush=True)

    model = CrossNet(
        num_classes=args.num_classes,
        conditioned=not args.no_conditioned,
        pretrained=False,
    ).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f'[*] Loaded step={ck.get("step", "?")}  epoch={ck.get("epoch", "?")}')
    if 'val_miou' in ck:
        print(f'    stored val_miou in ckpt: {ck["val_miou"]:.4f}')

    val_ds = CVPRDataset(args.val_csv)
    if args.max_samples > 0:
        val_ds = Subset(val_ds, range(min(args.max_samples, len(val_ds))))
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'[*] Val images : {len(val_ds)}')

    metrics = evaluate_crossnet(
        model, val_dl, device,
        num_classes=args.num_classes,
        use_amp=use_amp,
    )
    print()
    print(format_metrics(metrics, CLASS_NAMES))
    print()

    os.makedirs(args.dump_dir, exist_ok=True)
    row = metrics_row(metrics, epoch=int(ck.get('epoch', -1)), step=int(ck.get('step', -1)))
    csv_path = os.path.join(args.dump_dir, 'val_metrics.csv')
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    json_path = os.path.join(args.dump_dir, 'val_metrics_latest.json')
    payload = {
        'ckpt': os.path.abspath(ckpt),
        'n': metrics['n'],
        'miou_full': metrics['full']['miou'],
        'miou_native': metrics['low']['miou'],
        'acc_full': metrics['full']['acc'],
        'iou_full': {
            name: metrics['full']['iou'][i].item()
            for i, name in enumerate(CLASS_NAMES)
        },
    }
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'[*] Wrote {csv_path}')
    print(f'[*] Wrote {json_path}')

    if args.save_n > 0:
        n_saved = 0
        for batch in val_dl:
            aerial, ground, label = batch[0], batch[1], batch[2]
            aerial = aerial.to(device)
            with torch.inference_mode():
                la, lg = model(aerial)
            for i in range(aerial.size(0)):
                save_eval_montage(
                    aerial[i:i + 1], ground[i:i + 1], label[i:i + 1],
                    la[i:i + 1], lg[i:i + 1],
                    step=n_saved, dump_dir=args.dump_dir,
                    prefix='val', num_classes=args.num_classes,
                )
                n_saved += 1
                if n_saved >= args.save_n:
                    break
            if n_saved >= args.save_n:
                break
        print(f'[*] Saved {n_saved} montages -> {args.dump_dir}')


if __name__ == '__main__':
    main()
