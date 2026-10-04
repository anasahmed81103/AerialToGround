"""Append polar_full_classweighted test metrics to experiments/results_table.json."""
import json
import os

EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
run_json = os.path.join(EXP, 'phase2', 'runs', 'polar_full_classweighted', 'results.json')
table_path = os.path.join(EXP, 'results_table.json')

data = json.load(open(run_json))
t = data['test_full_res']
entry = {
    'name': 'polar_full_classweighted',
    'run_dir': 'experiments/phase2/runs/polar_full_classweighted/',
    'native_out_grid': [32, 160],
    'class_weights': data.get('class_weights'),
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
ab['polar_full_classweighted'] = {
    'miou': t['miou'],
    'class_weights': data.get('class_weights'),
    'iou': t['iou'],
}
phase2['polar_full_classweighted'] = entry
json.dump(table, open(table_path, 'w'), indent=2)
print(f'[*] Updated {table_path}')
