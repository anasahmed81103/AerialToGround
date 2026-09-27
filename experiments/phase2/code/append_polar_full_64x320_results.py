"""Append polar_full_64x320 test metrics to experiments/results_table.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
run_json = os.path.join(ROOT, 'phase2', 'runs', 'polar_full_64x320', 'results.json')
table_path = os.path.join(ROOT, 'results_table.json')

data = json.load(open(run_json))
t = data['test_full_res']
entry = {
    'name': 'polar_full_64x320',
    'run_dir': 'experiments/phase2/runs/polar_full_64x320/',
    'native_out_grid': [64, 320],
    'test_protocol': 'bilinear align_corners=True upsample native logits to 224x1232',
    'training': {
        'epochs': 8,
        'batch_size': 16,
        'params_trainable': data.get('params_trainable'),
        'peak_vram_gb': data.get('peak_vram_gb'),
        'train_minutes': data.get('train_minutes'),
    },
    'test_full8884_v2': {
        'best_epoch': t['best_epoch'],
        'miou_full': t['miou'],
        'pixel_acc_full': t['acc'],
        'iou': {k: t['iou'][k] for k in ('sky', 'vegetation', 'road', 'building')},
    },
}

table = json.load(open(table_path))
phase2 = table.setdefault('phase2', {})
ab = phase2.setdefault('ablations_test_full8884_v2', {})
ab['polar_full_64x320'] = {
    'miou': t['miou'],
    'native_grid': '64x320',
    'iou': t['iou'],
}
phase2['polar_full_64x320'] = entry
json.dump(table, open(table_path, 'w'), indent=2)
print(f'[*] Updated {table_path}')
