#!/usr/bin/env python
"""
tools.py -- helper for paper/main.tex (Aerial2Ground).

    python paper/tools.py facts   [--json]          verified facts sheet from the run artefacts
    python paper/tools.py bib     [--offline]       arXiv metadata -> \\bibitem entries (+ CrossViewDiff table check)
    python paper/tools.py figures [--no-regen] [--only f1,f3,f4,f5,f6,app,uncurated]
    python paper/tools.py audit                     every number in main.tex vs the facts whitelist (+ style/build checks)

Sources (repo root): experiments/results_table.json, experiments/phase2/runs/*/{results,config}.json,
final_eval_results/experiments/final_eval/ (full A-E metrics, per-image npz, bootstrap_ci.json, sample PNGs),
logs/*.txt, bootstrap_ci.py, cache/dinov2s_reg_336/meta.json, cvpr_*_v2.csv, model code for architecture constants.
experiments/final_eval/ holds only a 64-image smoke run and is rejected automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict

PAPER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER)
FIG = os.path.join(PAPER, 'figures')
TEX = os.path.join(PAPER, 'main.tex')

FINAL_EVAL_CANDIDATES = [
    os.path.join('experiments', 'final_eval'),
    os.path.join('final_eval_results', 'experiments', 'final_eval'),
]
CONFIG_KEYS = ['A', 'B', 'C', 'D', 'E']
CONFIG_SLUGS = {
    'A': 'pix2pix_random_d_gt', 'B': 'spade_v2_gt', 'C': 'spade_v2_pred',
    'D': 'spade_aerial_v2_gt', 'E': 'spade_aerial_v2_pred',
}
CLASSES = [('sky', 'Sky'), ('vegetation', 'Veg'), ('road', 'Road'), ('building', 'Bldg')]

STAGE1_RUNS = OrderedDict([
    ('Full', 'polar_full'), ('PriorNone', 'polar_prior_none'), ('PriorFixed', 'polar_prior_fixed'),
    ('NoOffsets', 'polar_no_offsets'), ('Dense', 'polar_dense_mapping'), ('Crop', 'polar_crop224'),
    ('Vgg', 'polar_vgg16'), ('HiRes', 'polar_full_64x320'), ('ClassW', 'polar_full_classweighted'),
])
GAN_LOGS = OrderedDict([
    ('Diag', 'gan_v2_diag.txt'), ('DiagTwo', 'gan_v2_diag2.txt'), ('DiagThree', 'gan_v2_diag3.txt'),
    ('Retrain', 'gan_v2_retrain.txt'), ('RandomD', 'gan_v2_random_d.txt'),
    ('Spade', 'spade_v2.txt'), ('SpadeAerial', 'spade_aerial_v2.txt'),
])
# Runs drawn in the discriminator-dynamics figure (F6).
F6_RUNS = [('Retrain', 'Pix2Pix, warm-started D (AMP)'),
           ('DiagTwo', 'Pix2Pix, warm-started D (FP32 check)'),
           ('RandomD', 'Pix2Pix, random-init D (Config A)'),
           ('Spade', 'SPADE, random-init D (Configs B, C)')]

# Table 1 of CrossViewDiff (arXiv 2408.14765), CVUSA columns. Verified against the PDF text by `bib`.
PUBLISHED = OrderedDict([
    ('CDTE', dict(ssim=0.283, sd=15.24, psnr=13.815, fid=28.35, kid=0.028)),
    ('S2SP', dict(ssim=0.319, sd=15.73, psnr=13.689, fid=27.31, kid=0.021)),
    ('Sat2Density', dict(ssim=0.339, sd=15.73, psnr=14.229, fid=41.43, kid=0.036)),
    ('CrossViewDiff', dict(ssim=0.371, sd=16.31, psnr=12.000, fid=23.67, kid=0.018)),
])
PUBLISHED_ROW_LABEL = {'CDTE': 'CDTE Toker', 'S2SP': 'S2SP Shi', 'Sat2Density': 'Sat2Density Qian',
                       'CrossViewDiff': 'Ours'}

# arXiv id -> (bib key, expected title words, venue/year to print, short role)
BIB = OrderedDict([
    ('1612.02709', ('zhai2017crossnet', 'Ground-Level Scene Layout Aerial', 'CVPR', 2017)),
    ('1510.03743', ('workman2015cvusa', 'Geolocalization Aerial Reference', 'ICCV', 2015)),
    # The brief listed 1803.03831, which the arXiv API resolves to an unrelated paper; 1803.03396 was found
    # by the SEARCH_ON_MISMATCH title+author query below and is used instead (flagged in main.tex with a \todo).
    ('1803.03396', ('regmi2018crossview', 'Cross-View Image Synthesis', 'CVPR', 2018)),
    ('1904.06807', ('tang2019selectiongan', 'Attention Selection GAN', 'CVPR', 2019)),
    ('2103.06818', ('toker2021cdte', 'Coming Down to Earth', 'CVPR', 2021)),
    ('2103.01623', ('shi2022s2sp', 'Street-View Panorama Synthesis', 'IEEE TPAMI', 2022)),
    ('2303.14672', ('qian2023sat2density', 'Sat2Density', 'ICCV', 2023)),
    ('2408.14765', ('li2024crossviewdiff', 'CrossViewDiff', 'arXiv preprint', 2024)),
    ('2505.17001', ('qian2025sat2density', 'Satellite', 'arXiv preprint', 2025)),
    ('1611.07004', ('isola2017pix2pix', 'Image-to-Image Translation', 'CVPR', 2017)),
    ('1903.07291', ('park2019spade', 'Spatially-Adaptive Normalization', 'CVPR', 2019)),
    ('2304.07193', ('oquab2023dinov2', 'DINOv2', 'arXiv preprint', 2023)),
    ('2309.16588', ('darcet2023registers', 'Registers', 'arXiv preprint', 2023)),
    ('2105.15203', ('xie2021segformer', 'SegFormer', 'NeurIPS', 2021)),
    ('1801.03924', ('zhang2018lpips', 'Unreasonable Effectiveness', 'CVPR', 2018)),
    ('1706.08500', ('heusel2017fid', 'Two Time-Scale Update Rule', 'NeurIPS', 2017)),
    ('2104.11222', ('parmar2022cleanfid', 'Aliased Resizing', 'CVPR', 2022)),
    ('1801.01401', ('binkowski2018kid', 'Demystifying MMD GANs', 'ICLR', 2018)),
    ('1603.08155', ('johnson2016perceptual', 'Perceptual Losses', 'ECCV', 2016)),
    ('1611.04076', ('mao2017lsgan', 'Least Squares Generative Adversarial', 'ICCV', 2017)),
    ('1708.02002', ('lin2017focal', 'Focal Loss', 'ICCV', 2017)),
    ('2302.05543', ('zhang2023controlnet', 'Conditional Control', 'ICCV', 2023)),
])

OKABE_ITO = {
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7',
}
# Layout colours used in every figure: sky, vegetation, road, building.
LAYOUT_RGB = [(0x56, 0xB4, 0xE9), (0x00, 0x9E, 0x73), (0xE6, 0x9F, 0x00), (0xCC, 0x79, 0xA7)]
# Palette of the repository's own showcase images (semantic_metrics.CLASS_COLORS).
REPO_RGB = [(255, 64, 64), (64, 200, 64), (64, 64, 255), (255, 220, 30)]
LM_FONT = r'C:\Users\Anas\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf'


# ============================================================================ helpers

def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def load_json(*parts: str) -> dict:
    with open(p(*parts), encoding='utf-8') as f:
        return json.load(f)


def read_text(*parts: str) -> str:
    with open(p(*parts), encoding='utf-8', errors='replace') as f:
        return f.read()


def find_final_eval_dir() -> str:
    for cand in FINAL_EVAL_CANDIDATES:
        ok = os.path.isfile(p(cand, 'bootstrap_ci.json'))
        for slug in CONFIG_SLUGS.values():
            mp = p(cand, f'{slug}_metrics.json')
            if not ok or not os.path.isfile(mp):
                ok = False
                break
            m = load_json(cand, f'{slug}_metrics.json')
            if m.get('smoke') or m.get('n_images') != 8884 or 'fid_clean' not in m:
                ok = False
                break
        if ok:
            return cand
        print(f'[i] skipping {cand}: missing files or smoke / partial run', file=sys.stderr)
    raise SystemExit('No complete final_eval directory found')


def parse_gan_log(fname: str) -> dict:
    text = read_text('logs', fname)
    out: dict = {'file': fname}
    gpus = re.findall(r'\[\*\] GPU\s*:\s*(.+?)\s+\(', text)
    out['gpu'] = gpus[-1].replace('NVIDIA ', '').replace('GeForce ', '').strip() if gpus else None
    amps = re.findall(r'\[\*\] AMP\s*:\s*(ON|OFF)', text)
    out['amp'] = amps[-1] if amps else None
    gs = re.findall(r'G updates / D\s*:\s*(\d+)', text)
    out['g_steps'] = int(gs[-1]) if gs else None
    out['warm_d'] = bool(re.search(r'Resumed D (weights only|from checkpoint)', text))
    out['random_d'] = 'Discriminator: random init' in text
    gp = re.findall(r'Generator params\s*:\s*([\d.]+)\s*M', text)
    out['g_params_m'] = float(gp[-1]) if gp else None
    dp_ = re.findall(r'Discriminator params\s*:\s*([\d.]+)\s*M', text)
    out['d_params_m'] = float(dp_[-1]) if dp_ else None

    def last(pattern, cast=float):
        v = re.findall(pattern, text, flags=re.M)
        return cast(v[-1]) if v else None

    out['lambda_l1'] = last(r'^\s*lambda_l1\s+([\d.]+)')
    out['lambda_perc'] = last(r'VGG perceptual loss weight:\s*([\d.]+)')
    out['label_smooth'] = last(r'^\s*label_smooth\s+([\d.]+)')
    out['epochs_cfg'] = last(r'^\s*epochs\s+(\d+)', int)
    out['batch_size'] = last(r'^\s*batch_size\s+(\d+)', int)
    out['lr'] = last(r'^\s*lr\s+([\d.eE-]+)')
    out['save_every'] = last(r'^\s*save_every\s+(\d+)', int)
    out['steps_per_epoch'] = last(r'Steps per epoch\s*:\s*(\d+)', int)
    out['resumed_g_step'] = last(r'Resumed G from step (\d+)', int)
    hb = re.search(r'D_loss in \[([\d.]+)\D+?([\d.]+)\]', text)
    out['healthy_d'] = (float(hb.group(1)), float(hb.group(2))) if hb else None

    epochs = OrderedDict()
    for m in re.finditer(r'\[epoch (\d+) done\]\s+avg G=([\d.]+)\s+avg D=([\d.]+)\s+time=([\d.]+) min\s+lr=([\d.e+-]+)',
                         text):
        epochs[int(m.group(1))] = {'avg_g': float(m.group(2)), 'avg_d': float(m.group(3)),
                                   'min': float(m.group(4)), 'lr': float(m.group(5))}
    out['epochs'] = epochs

    steps = OrderedDict()
    for m in re.finditer(r'\[epoch (\d+)\] step (\d+)\s+G=([\d.]+) \(adv=([\d.]+) l1=([\d.]+)(?: perc=([\d.]+))?\)'
                         r'\s+D=([\d.]+)\[([^\]]*)\]', text):
        steps[int(m.group(2))] = {
            'epoch': int(m.group(1)), 'g': float(m.group(3)), 'adv': float(m.group(4)), 'l1': float(m.group(5)),
            'perc': float(m.group(6)) if m.group(6) else None, 'd': float(m.group(7)),
            'weak': m.group(8) == 'D_weak',
        }
    out['steps'] = steps
    out['saved'] = [int(s) for s in re.findall(r'\[ckpt\] saved.*?step(\d+)', text)]
    out['complete'] = 'GAN training complete' in text

    diag = re.findall(r'\[D_diag run_step\s+(\d+) global (\d+)\].*?amp=(\w+) real\(mean/min/max\)=([-\d.]+)/.*?'
                      r'fake\(mean/min/max\)=([-\d.]+)/', text)
    out['diag'] = [(int(a), int(b), c == 'True', float(r), float(f_)) for a, b, c, r, f_ in diag]
    return out


# ============================================================================ facts

class Facts:
    """name -> (value, source, dp, mindp). dp = display decimals (None for exact ints / strings)."""

    def __init__(self):
        self.d: 'OrderedDict[str, tuple]' = OrderedDict()
        self.warnings: list[str] = []

    def add(self, name, value, src, dp=None, mindp=None, pct=False):
        if name in self.d:
            raise ValueError(f'duplicate fact {name}')
        if mindp is None and dp is not None:
            mindp = 0 if (isinstance(value, (int, float)) and abs(value) >= 100) else min(dp, 1)
        self.d[name] = (value, src, dp, mindp, pct)
        return value

    def __getitem__(self, k):
        return self.d[k][0]

    def warn(self, msg):
        self.warnings.append(msg)


def build_facts() -> Facts:
    F = Facts()
    rt = load_json('experiments', 'results_table.json')
    fe = find_final_eval_dir()
    F.add('FinalEvalDir', fe.replace(os.sep, '/'), 'auto-detected')
    met = {k: load_json(fe, f'{CONFIG_SLUGS[k]}_metrics.json') for k in CONFIG_KEYS}
    boot = load_json(fe, 'bootstrap_ci.json')
    s1 = {t: load_json('experiments', 'phase2', 'runs', r, 'results.json') for t, r in STAGE1_RUNS.items()}
    s1c = {t: load_json('experiments', 'phase2', 'runs', r, 'config.json') for t, r in STAGE1_RUNS.items()}
    p2 = rt['phase2']
    cross = load_json(*p2['fair_old_models_full8884_v2']['baseline_88830']['source_file'].split('/'))
    ph1 = load_json(*p2['fair_old_models_full8884_v2']['phase1_best']['source_file'].split('/'))
    logs = {t: parse_gan_log(fn) for t, fn in GAN_LOGS.items()}

    # ------------------------------------------------------------ data / protocol
    def nrows(fn):
        with open(p(fn), encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    F.add('NTrain', nrows('cvpr_train_v2.csv'), 'cvpr_train_v2.csv rows')
    F.add('NTest', nrows('cvpr_val_v2.csv'), 'cvpr_val_v2.csv rows')
    if F['NTest'] != 8884 or any(met[k]['n_images'] != 8884 for k in CONFIG_KEYS):
        F.warn('test split size != 8884')
    F.add('NPairsTotal', F['NTrain'] + F['NTest'], 'derived: train + test')
    F.add('NMinival', int(s1c['Full']['minival']), 'polar_full/config.json')
    F.add('NTrainFit', F['NTrain'] - F['NMinival'], 'derived: train - minival')
    F.add('MinivalSeed', int(s1c['Full']['seed']), 'polar_full/config.json')
    rl = rt['phase1']['relabel']
    F.add('RelabelPairs', rl['pairs'], 'results_table.phase1.relabel')
    F.add('RelabelMinutes', rl['minutes'], 'results_table.phase1.relabel', 1)
    F.add('RelabelHours', rl['minutes'] / 60, 'derived', 2)
    F.add('NClasses', len(rt['palette']), 'results_table.palette')
    bsrc = read_text('bootstrap_ci.py')
    F.add('NBoot', int(re.search(r'^N_BOOT\s*=\s*(\d+)', bsrc, re.M).group(1)), 'bootstrap_ci.py')
    F.add('BootSeed', int(re.search(r'^SEED\s*=\s*(\d+)', bsrc, re.M).group(1)), 'bootstrap_ci.py')
    F.add('CILevel', 95, 'bootstrap_ci.py: percentiles 2.5/97.5')
    F.add('CIPctLo', 2.5, 'bootstrap_ci.py', 1)
    F.add('CIPctHi', 97.5, 'bootstrap_ci.py', 1)
    F.add('NSamplesSaved', 24, 'final_eval.py KEEP_N')

    # ------------------------------------------------------------ Stage 1 setup
    c = s1c['Full']
    for k, name in (('epochs', 'SOneEpochs'), ('batch_size', 'SOneBatch'), ('warmup', 'SOneWarmup'),
                    ('dim', 'SOneDim'), ('radial_bins', 'SOneRadialBins'), ('out_h', 'SOneOutH'),
                    ('out_w', 'SOneOutW'), ('layers', 'SOneLayers'), ('heads', 'SOneHeads')):
        F.add(name, int(c[k]), 'polar_full/config.json')
    F.add('SOneLR', c['lr'], 'polar_full/config.json', 4)
    F.add('SOneWeightDecay', c['weight_decay'], 'polar_full/config.json', 2)
    F.add('SOneFocalGamma', int(c['focal_gamma']), 'polar_full/config.json')
    F.add('SOneHorizon', c['horizon'], 'polar_full/config.json', 1)
    F.add('SOneGradClip', 1.0, 'train_polar.py clip_grad_norm_', 1, 0)
    F.add('SOneDiceSmooth', 1.0, 'phase2_weighted_loss.dice_loss smooth', 1, 0)
    F.add('SOneMaxOffset', 0.1, 'polar_layout_model.PolarLayoutNet max_offset default', 1)
    F.add('SOnePriorSigma', 1.5, 'polar_layout_model.ground_plane_bias sigma_bins', 1)
    F.add('SOneResBlocks', 2, 'polar_layout_model: proj and refine use 2 ResBlocks each')
    F.add('SOneGroupNorm', 8, 'polar_layout_model GroupNorm(8, dim)')
    F.add('SOneFFMult', 2, 'polar_layout_model ff hidden = 2*dim')
    meta = load_json('cache', 'dinov2s_reg_336', 'meta.json')
    F.add('DinoInput', int(meta['size']), 'cache/dinov2s_reg_336/meta.json')
    F.add('DinoGrid', int(meta['grid']), 'cache/dinov2s_reg_336/meta.json')
    F.add('DinoDim', int(meta['dim']), 'cache/dinov2s_reg_336/meta.json')
    F.add('DinoPatch', 14, 'cache_dino_features.py stride for DINOv2')
    F.add('TileSize', 750, 'cache meta: full 750x750 tile')
    F.add('CropSize', 224, 'run_phase2_ablations2.ps1: --crop 224 --size 224')
    F.add('CropGrid', 16, 'run_phase2_ablations2.ps1 comment: 16x16 tokens (224/14)')
    F.add('VggGrid', 21, 'run_phase2_ablations2.ps1 comment: 21x21 (336/16)')
    F.add('VggDim', 512, 'cache_dino_features.VGGConv5: 512 ch')
    F.add('VggStride', 16, 'cache_dino_features.VGGConv5: stride 16')
    ev = re.search(r'(\d+)x(\d+)', p2['protocol'].split('report on')[-1])
    F.add('SOneEvalH', int(ev.group(1)), 'results_table.phase2.protocol')
    F.add('SOneEvalW', int(ev.group(2)), 'results_table.phase2.protocol')
    F.add('SOneHiResOutH', int(s1c['HiRes']['out_h']), 'polar_full_64x320/config.json')
    F.add('SOneHiResOutW', int(s1c['HiRes']['out_w']), 'polar_full_64x320/config.json')
    F.add('SOneHiResBatch', int(s1c['HiRes']['batch_size']), 'polar_full_64x320/config.json')
    cw = s1['ClassW'].get('class_weights') or p2['polar_full_classweighted']['class_weights']
    for cls, ct in CLASSES:
        F.add(f'ClassW{ct}', cw[cls], 'polar_full_classweighted/results.json class_weights', 3, 2)
    F.add('SOneGpu', p2['training']['gpu'], 'results_table.phase2.training.gpu')
    F.add('SOneVramMax', 4, 'results_table.phase2.training.gpu = Quadro P1000 4GB')
    F.add('SOneHiResGpu', p2['polar_full_64x320']['training']['gpu'], 'results_table')

    # ------------------------------------------------------------ Stage 1 results
    for t, r in s1.items():
        te = r['test_full_res']
        assert te['n'] == 8884, t
        src = f'{STAGE1_RUNS[t]}/results.json'
        F.add(f'SOne{t}Miou', te['miou'], src, 3)
        F.add(f'SOne{t}Acc', te['acc'], src, 3)
        for cls, ct in CLASSES:
            F.add(f'SOne{t}{ct}', te['iou'][cls], src, 3)
        F.add(f'SOne{t}Params', r['params_trainable'], src)
        F.add(f'SOne{t}ParamsM', r['params_trainable'] / 1e6, src, 2)
        F.add(f'SOne{t}Minutes', r['train_minutes'], src, 1, 0)
        F.add(f'SOne{t}Hours', r['train_minutes'] / 60, src, 2)
        F.add(f'SOne{t}VramGB', r['peak_vram_gb'], src, 2)
        F.add(f'SOne{t}BestEpoch', int(te['best_epoch']), src)
        F.add(f'SOne{t}MinivalMiou', r['best_minival_miou'], src, 3)
        F.add(f'SOne{t}Amp', bool(s1c[t].get('amp')), f'{STAGE1_RUNS[t]}/config.json')
    for t, r, src in (('CrossNet', cross, 'baseline eval_v2_labels_full8884'),
                      ('PhaseOne', ph1, 'phase1 eval_v2_labels_full8884')):
        assert r['n'] == 8884
        F.add(f'SOne{t}Miou', r['miou_full'], src, 3)
        F.add(f'SOne{t}Acc', r['acc_full'], src, 3)
        for cls, ct in CLASSES:
            F.add(f'SOne{t}{ct}', r['iou_full'][cls], src, 3)
    old = rt['baseline']['eval_on_old_labels_full_val']
    F.add('SOneCrossNetOldLabelsMiou', old['miou_full'], 'results_table.baseline.eval_on_old_labels_full_val', 3)
    F.add('SOneCrossNetGrid', '8x40', 'results_table.baseline.training.loss')
    F.add('SOneCrossNetGridH', 8, 'results_table.baseline: 8x40')
    F.add('SOneCrossNetGridW', 40, 'results_table.baseline: 8x40')
    F.add('SOneCrossNetEpochs', rt['baseline']['training']['epoch'], 'results_table.baseline.training')
    F.add('SOneCrossNetStep', rt['baseline']['training']['step'], 'results_table.baseline.training')
    ph1t = rt['phase1']['training']
    F.add('PhaseOneEpochs', ph1t['epochs'], 'results_table.phase1.training')
    F.add('PhaseOneSteps', ph1t['steps'], 'results_table.phase1.training')
    F.add('PhaseOneBatch', ph1t['batch_size'], 'results_table.phase1.training')
    F.add('PhaseOneLR', ph1t['lr'], 'results_table.phase1.training', 3)
    F.add('PhaseOneAux', ph1t['aux_weight'], 'results_table.phase1.training', 1)
    F.add('PhaseOneValN', ph1t['val_during_train_n'], 'results_table.phase1.training')

    fm = s1['Full']['test_full_res']['miou']
    full = s1['Full']['test_full_res']

    def mi(t):
        return s1[t]['test_full_res']['miou']
    F.add('SOneDeltaCrossNet', fm - cross['miou_full'], 'derived: polar_full - original CrossNet', 3)
    F.add('SOneDeltaPhaseOne', fm - ph1['miou_full'], 'derived: polar_full - retrained CrossNet', 3)
    F.add('SOneDeltaPhaseOneOverCrossNet', ph1['miou_full'] - cross['miou_full'], 'derived', 3)
    F.add('SOneGainDinoOverVgg', fm - mi('Vgg'), 'derived', 3)
    F.add('SOneGainFullTileOverCrop', fm - mi('Crop'), 'derived', 3)
    F.add('SOneGainPolarOverDense', fm - mi('Dense'), 'derived', 4)
    F.add('SOneGainLearnedOverNone', fm - mi('PriorNone'), 'derived', 4)
    F.add('SOneGainLearnedOverFixed', fm - mi('PriorFixed'), 'derived', 4)
    F.add('SOneGainOffsets', fm - mi('NoOffsets'), 'derived', 4)
    F.add('SOneGainLowResHead', fm - mi('HiRes'), 'derived', 4)
    F.add('SOneClassWDeltaMiou', mi('ClassW') - fm, 'derived', 3)
    F.add('SOneClassWDeltaBldg', s1['ClassW']['test_full_res']['iou']['building'] - full['iou']['building'],
          'derived', 3)
    for t in ('PriorNone', 'PriorFixed', 'NoOffsets', 'Dense', 'Crop', 'Vgg', 'HiRes', 'ClassW'):
        F.add(f'SOneAbl{t}Delta', mi(t) - fm, 'derived: ablation - reference', 3)
    nulls = [mi(t) - fm for t in ('PriorNone', 'PriorFixed', 'NoOffsets', 'Dense', 'HiRes')]
    F.add('SOneNullSpread', max(abs(x) for x in nulls), 'derived: max |delta| over null ablations', 3)
    F.add('SOneDenseParamsRatio', s1['Dense']['params_trainable'] / s1['Full']['params_trainable'], 'derived', 1)
    for cls, ct in CLASSES:
        F.add(f'SOneDeltaPhaseOne{ct}', full['iou'][cls] - ph1['iou_full'][cls], 'derived', 3)
        F.add(f'SOneDeltaCrossNet{ct}', full['iou'][cls] - cross['iou_full'][cls], 'derived', 3)
    abl_rt = p2['ablations_test_full8884_v2']
    for t, key in (('Full', 'polar_full (learned prior + offsets)'), ('PriorNone', 'polar_prior_none'),
                   ('PriorFixed', 'polar_prior_fixed'), ('NoOffsets', 'polar_no_offsets'),
                   ('Dense', 'polar_dense_mapping'), ('Crop', 'polar_crop224'), ('Vgg', 'polar_vgg16'),
                   ('HiRes', 'polar_full_64x320'), ('ClassW', 'polar_full_classweighted')):
        if abs(mi(t) - abl_rt[key]['miou']) > 6e-4:
            F.warn(f'Stage 1 {t}: results.json {mi(t):.4f} vs results_table {abl_rt[key]["miou"]:.4f}')

    # ------------------------------------------------------------ Stage 2 setup (logs)
    sp, sa, rd, rr = logs['Spade'], logs['SpadeAerial'], logs['RandomD'], logs['Retrain']
    F.add('SpadeLambdaLOne', int(sp['lambda_l1']), 'logs/spade_v2.txt')
    F.add('SpadeLambdaPerc', int(sp['lambda_perc']), 'logs/spade_v2.txt')
    F.add('PixLambdaLOne', int(rd['lambda_l1']), 'logs/gan_v2_random_d.txt')
    F.add('GanLR', sp['lr'], 'logs/spade_v2.txt', 4)
    F.add('GanBetaOne', 0.5, 'train_gan_pix2pix.py Adam betas', 1)
    F.add('GanBetaTwo', 0.999, 'train_gan_pix2pix.py Adam betas', 3)
    F.add('GanBatch', sp['batch_size'], 'logs/spade_v2.txt')
    F.add('GanEpochs', sp['epochs_cfg'], 'logs/spade_v2.txt')
    F.add('GanLabelSmooth', sp['label_smooth'], 'logs/spade_v2.txt', 1)
    F.add('GanGSteps', sp['g_steps'], 'logs/spade_v2.txt')
    F.add('GanGStepsRetrain', rr['g_steps'], 'logs/gan_v2_retrain.txt')
    F.add('GanStepsPerEpoch', sp['steps_per_epoch'], 'logs/spade_v2.txt')
    F.add('GanTotalSteps', sp['steps_per_epoch'] * sp['epochs_cfg'], 'derived: steps/epoch * epochs')
    F.add('GenParamsPix', rd['g_params_m'], 'logs/gan_v2_random_d.txt', 2, 1)
    F.add('GenParamsSpade', sp['g_params_m'], 'logs/spade_v2.txt', 2, 1)
    F.add('GenParamsSpadeAerial', sa['g_params_m'], 'logs/spade_aerial_v2.txt', 2, 1)
    F.add('DiscParams', sp['d_params_m'], 'logs/spade_v2.txt', 2)
    F.add('GanImgH', 256, 'final_eval.py --img_h')
    F.add('GanImgW', 512, 'final_eval.py --img_w')
    F.add('GanNgf', 64, 'logs: ngf 64')
    F.add('SpadeLatentH', 8, 'spade_model.py: H/32 for 256')
    F.add('SpadeLatentW', 16, 'spade_model.py: W/32 for 512')
    F.add('SpadeUpLayers', 5, 'spade_model.py num_up_layers')
    F.add('SpadeHidden', 128, 'spade_model.SPADE hidden')
    F.add('SpadeCondAerialCh', 7, 'spade_model: 4 layout + 3 aerial channels')
    F.add('PercLayers', 4, 'gan_perceptual_loss.py: relu1_1..relu4_1')
    F.add('LsganHalf', 0.5, 'train_gan_pix2pix.discriminator_loss factor', 1)
    F.add('HealthyDLo', sp['healthy_d'][0], 'training log banner', 2)
    F.add('HealthyDHi', sp['healthy_d'][1], 'training log banner', 2)
    F.add('PixWarmStartStep', rd['resumed_g_step'], 'logs/gan_v2_random_d.txt: Resumed G from step')
    F.add('PixWarmStartEpoch', 38, 'logs: Resumed G from step 197766, epoch 38')
    F.add('SpadeAerialResumeStep', sa['resumed_g_step'], 'logs/spade_aerial_v2.txt')
    F.add('SpadeAerialResumeEpoch', min(sa['epochs']) - 1, 'logs/spade_aerial_v2.txt (first timed epoch - 1)')
    F.add('SpadeAerialFirstLoggedEpoch', min(sa['epochs']), 'logs/spade_aerial_v2.txt')
    F.add('SpadeAerialLastStep', max(sa['steps']), 'logs/spade_aerial_v2.txt last logged step')
    F.add('SpadeAerialSaveEvery', sa['save_every'], 'logs/spade_aerial_v2.txt (resumed segment)')
    F.add('SpadeAerialLastSaved', max(sa['saved']), 'logs/spade_aerial_v2.txt last [ckpt] saved')
    F.add('SpadeLastSaved', max(sp['saved']), 'logs/spade_v2.txt last [ckpt] saved')
    F.add('PixLastSaved', max(rd['saved']), 'logs/gan_v2_random_d.txt last [ckpt] saved')
    F.add('SpadeDecayStartEpoch', min(e for e, v in sp['epochs'].items() if v['lr'] < sp['lr']) - 1,
          'logs/spade_v2.txt: last epoch at full LR')
    F.add('SpadeFinalLR', sp['epochs'][max(sp['epochs'])]['lr'], 'logs/spade_v2.txt', 1, 0)
    F.add('SpadeAerialFinalLR', sa['epochs'][max(sa['epochs'])]['lr'], 'logs/spade_aerial_v2.txt', 4)
    F.add('PixFinalLR', rd['epochs'][max(rd['epochs'])]['lr'], 'logs/gan_v2_random_d.txt', 4)
    for k, lg in (('A', rd), ('B', sp), ('D', sa)):
        step = met[k]['generator_step']
        if max(lg['saved']) != step:
            F.warn(f'Config {k}: evaluated step {step} != last saved {max(lg["saved"])}')
    F.add('ReportedIsLastSaved', 'yes', 'checked: A, B/C, D/E generator_step == last [ckpt] saved in log')

    # ------------------------------------------------------------ Stage 2 results (final_eval)
    pc = boot['per_config']
    for k in CONFIG_KEYS:
        m, b = met[k], pc[k]
        src = f'final_eval/{CONFIG_SLUGS[k]}_metrics.json + bootstrap_ci.json'
        for mt, tok, dp in (('ssim', 'Ssim', 3), ('psnr', 'Psnr', 2), ('lpips', 'Lpips', 3), ('sd', 'Sharp', 0)):
            if abs(b[mt]['mean'] - m[f'mean_{mt}']) > 1e-6:
                F.warn(f'Config {k} {mt}: bootstrap mean != metrics mean')
            F.add(f'Cfg{k}{tok}', m[f'mean_{mt}'], src, dp)
            F.add(f'Cfg{k}{tok}Lo', b[mt]['ci95_low'], src, dp)
            F.add(f'Cfg{k}{tok}Hi', b[mt]['ci95_high'], src, dp)
        F.add(f'Cfg{k}Fid', m['fid_clean'], src, 1)
        F.add(f'Cfg{k}Kid', m['kid_clean'], src, 3)
        F.add(f'Cfg{k}EvalMin', m['elapsed_sec'] / 60, src, 1, 0)
        F.add(f'Cfg{k}Step', m['generator_step'], src)
    rt_fe = rt['final_eval_publication']['configs']
    for k in CONFIG_KEYS:
        for mk, key, tol in (('fid_clean', 'fid_clean', 0.051), ('mean_lpips', 'lpips', 0.0051),
                             ('mean_ssim', 'ssim', 0.0051), ('kid_clean', 'kid', 0.0051)):
            if abs(met[k][mk] - rt_fe[k][key]) > tol:
                F.warn(f'Config {k} {key}: metrics {met[k][mk]} vs results_table {rt_fe[k][key]}')
    pr = boot['paired']
    for pair, tok in (('D_minus_B', 'DB'), ('E_minus_C', 'EC')):
        for mt, mtok, dp in (('ssim', 'Ssim', 4), ('psnr', 'Psnr', 2), ('lpips', 'Lpips', 3)):
            d = pr[pair][mt]
            F.add(f'Pair{tok}{mtok}', d['mean_diff'], f'bootstrap_ci.json paired {pair}', dp)
            F.add(f'Pair{tok}{mtok}Lo', d['ci95_low'], f'bootstrap_ci.json paired {pair}', dp)
            F.add(f'Pair{tok}{mtok}Hi', d['ci95_high'], f'bootstrap_ci.json paired {pair}', dp)
            F.add(f'Pair{tok}{mtok}ExclZero', 'yes' if (d['ci95_low'] > 0 or d['ci95_high'] < 0) else 'no',
                  'derived')
    fid = {k: met[k]['fid_clean'] for k in CONFIG_KEYS}
    kid = {k: met[k]['kid_clean'] for k in CONFIG_KEYS}
    lp = {k: met[k]['mean_lpips'] for k in CONFIG_KEYS}
    ps = {k: met[k]['mean_psnr'] for k in CONFIG_KEYS}
    ss = {k: met[k]['mean_ssim'] for k in CONFIG_KEYS}
    F.add('FidOracleGap', fid['E'] - fid['D'], 'derived: E - D', 1)
    F.add('FidOracleGapNoAerial', fid['C'] - fid['B'], 'derived: C - B', 1)
    F.add('KidOracleGap', kid['E'] - kid['D'], 'derived: E - D', 3)
    F.add('LpipsOracleGap', lp['E'] - lp['D'], 'derived: E - D', 3)
    F.add('PsnrOracleGap', ps['D'] - ps['E'], 'derived: D - E', 2)
    F.add('SsimOracleGap', ss['D'] - ss['E'], 'derived: D - E', 3)
    F.add('FidAerialGainOracle', fid['B'] - fid['D'], 'derived: B - D', 1)
    F.add('FidAerialGainPred', fid['C'] - fid['E'], 'derived: C - E', 1)
    F.add('FidPairDB', fid['D'] - fid['B'], 'derived: D - B', 1)
    F.add('FidPairEC', fid['E'] - fid['C'], 'derived: E - C', 1)
    F.add('KidPairDB', kid['D'] - kid['B'], 'derived: D - B', 3)
    F.add('KidPairEC', kid['E'] - kid['C'], 'derived: E - C', 3)
    F.add('FidSpadeGainOverPix', fid['A'] - fid['B'], 'derived: A - B', 1)
    F.add('LpipsSpadeGainOverPix', lp['A'] - lp['B'], 'derived: A - B', 3)
    F.add('SsimPixMinusSpade', ss['A'] - ss['B'], 'derived: A - B', 3)
    F.add('FidGapToAerialRatio', (fid['E'] - fid['D']) / (fid['B'] - fid['D']), 'derived', 1)
    F.add('LpipsGapToAerialRatio', (lp['E'] - lp['D']) / (lp['B'] - lp['D']), 'derived', 1)

    # per-image selection for the F4 grid (Config E LPIPS quantiles)
    import numpy as np
    npz = {k: np.load(p(fe, f'{CONFIG_SLUGS[k]}_per_image.npz')) for k in CONFIG_KEYS}
    e_idx, e_lp = npz['E']['index'], npz['E']['lpips']
    order = np.argsort(e_lp, kind='stable')
    n = len(order)
    sel = OrderedDict()
    for lab, q in (('Best', 0.0), ('QOne', 0.25), ('Median', 0.5), ('QThree', 0.75), ('Worst', 1.0)):
        r = int(round(q * (n - 1)))
        sel[lab] = (int(e_idx[order[r]]), float(e_lp[order[r]]), r)
        F.add(f'SelIdx{lab}', sel[lab][0], 'npz: Config E LPIPS rank selection')
        F.add(f'SelLpips{lab}', sel[lab][1], 'npz: Config E per-image LPIPS', 3)
    for j, lab in enumerate(('FailTwo', 'FailThree', 'FailFour')):
        r = n - 2 - j
        F.add(f'SelIdx{lab}', int(e_idx[order[r]]), 'npz: Config E LPIPS rank n-2-j')
        F.add(f'SelLpips{lab}', float(e_lp[order[r]]), 'npz: Config E per-image LPIPS', 3)
    pos_e = {int(i): j for j, i in enumerate(e_idx)}
    unc_idx = np.random.RandomState(0).choice(8884, 6, replace=False).astype(int)
    for k, i in enumerate(unc_idx):
        F.add(f'UncIdx{k}', int(i), 'fig_app_uncurated RandomState(0).choice(8884,6)')
        F.add(f'UncLpips{k}', float(e_lp[pos_e[int(i)]]), 'npz: Config E per-image LPIPS', 3)
    F.add('DistinctFigTestImages', 17, 'derived: distinct test indices in figure captions')
    F.add('SelQuantiles', [0, 25, 50, 75, 100], 'selection rule')
    F.add('QOne', 25, 'selection rule: 25th percentile')
    F.add('QThree', 75, 'selection rule: 75th percentile')
    lp500 = float(e_lp[list(e_idx).index(500)])
    F.add('TeaserIdx', 500, 'outputs/dataset_pipeline_test idx_00500')
    F.add('TeaserLpipsE', lp500, 'npz: Config E LPIPS at test index 500', 3)
    F.add('TeaserLpipsRankPct', 100.0 * float((e_lp < lp500).mean()), 'npz: percentile of index 500', 0, 0)
    F.add('PipelineSmokeIdx', [0, 100, 500, 1269, 2538, 3807], 'outputs/dataset_pipeline_test/README.txt')
    F.add('WarpIdxA', 1269, 'experiments/stage2_aerial/warp_check')
    F.add('WarpIdxB', 3807, 'experiments/stage2_aerial/warp_check')
    cal = load_json('experiments', 'phase2', 'results', 'geo_calibration.json')
    F.add('GeoCalN', int(cal['n']), 'geo_calibration.json n (geo_calibrate.py default --csv cvpr_train_v2.csv)')
    F.add('GeoCalRoadIou', float(cal['best']['road_iou_best']), 'geo_calibration.json best road IoU', 2)
    for k in CONFIG_KEYS:
        F.add(f'Cfg{k}LpipsMedian', float(np.median(npz[k]['lpips'])), 'npz', 3)
    F.add('FracImgsEBetterThanC', 100.0 * float((npz['E']['lpips'] < npz['C']['lpips']).mean()),
          'npz: % images with LPIPS(E) < LPIPS(C)', 1, 0)
    F.add('FracImgsDBetterThanB', 100.0 * float((npz['D']['lpips'] < npz['B']['lpips']).mean()),
          'npz: % images with LPIPS(D) < LPIPS(B)', 1, 0)

    # ------------------------------------------------------------ legacy FID (appendix)
    s2p, s2s = rt['stage2_pix2pix'], rt['stage2_spade']
    F.add('LegacyFidPixWarmD', s2p['gan_v2_retrain']['val_full8884_v2']['fid'], 'results_table', 1)
    F.add('LegacyFidPix', s2p['gan_v2_random_d']['val_full8884_v2']['fid'], 'results_table', 1)
    F.add('LegacyFidPixMidRun', s2p['gan_v2_random_d']['checkpoint_selection']['G_step0228846']['fid'],
          'results_table', 1)
    F.add('LegacyPixMidRunStep', 228846, 'results_table checkpoint_selection key')
    F.add('LegacyFidSpade', s2s['spade_v2']['val_full8884_v2']['fid'], 'results_table', 1)
    F.add('LegacyFidSpadeAerial', s2s['spade_aerial_v2']['val_full8884_v2']['fid'], 'results_table', 1)
    F.add('LegacySpadeStepsEvaluated', [31080, 32000, 33300], 'experiments/phase2/runs/spade_v2/val_metrics_*.json')
    for st in (31080, 32000, 33300):
        F.add(f'LegacySpadeStep{st}', st, 'spade_v2 val_metrics file name')
    F.add('LegacyLpipsPixWarmD', s2p['gan_v2_retrain']['val_full8884_v2']['lpips'], 'results_table', 3)
    F.add('LegacySsimPixWarmD', s2p['gan_v2_retrain']['val_full8884_v2']['ssim'], 'results_table', 3)
    F.add('LegacyRetrainStep', s2p['gan_v2_retrain']['val_full8884_v2']['generator_step'], 'results_table')

    # ------------------------------------------------------------ diagnostics (logs)
    for t, lg in logs.items():
        eps = lg['epochs']
        if eps:
            F.add(f'Log{t}FirstEpochAvgD', next(iter(eps.values()))['avg_d'], lg['file'], 4)
            F.add(f'Log{t}LastEpochAvgD', list(eps.values())[-1]['avg_d'], lg['file'], 4)
            F.add(f'Log{t}FirstEpoch', next(iter(eps)), lg['file'])
            F.add(f'Log{t}LastEpoch', list(eps)[-1], lg['file'])
            F.add(f'Log{t}EpochsLogged', len(eps), lg['file'])
        st = lg['steps']
        if st:
            F.add(f'Log{t}DWeakPct', 100 * sum(s['weak'] for s in st.values()) / len(st), lg['file'], 0, 0)
            F.add(f'Log{t}NStepLines', len(st), lg['file'])
            F.add(f'Log{t}MeanAdv', sum(s['adv'] for s in st.values()) / len(st), lg['file'], 2)
            late = [s for s in st.values() if s['epoch'] >= (next(iter(eps)) + 2 if eps else 3)]
            if late:
                F.add(f'Log{t}DWeakPctLate', 100 * sum(s['weak'] for s in late) / len(late), lg['file'], 0, 0)
            ep_first = min(s['epoch'] for s in st.values())
            first = [s for s in st.values() if s['epoch'] == ep_first]
            F.add(f'Log{t}DWeakPctFirstEpoch', 100 * sum(s['weak'] for s in first) / len(first), lg['file'], 0, 0)
        F.add(f'Log{t}GSteps', lg['g_steps'], lg['file'])
        F.add(f'Log{t}Amp', lg['amp'], lg['file'])
        F.add(f'Log{t}Gpu', lg['gpu'], lg['file'])
        F.add(f'Log{t}DInit', 'warm' if lg['warm_d'] and not lg['random_d'] else 'random', lg['file'])
    d2 = logs['DiagTwo']['diag']
    F.add('DiagTwoStepOneReal', d2[0][3], 'logs/gan_v2_diag2.txt D_diag run_step 1', 3)
    F.add('DiagTwoStepOneFake', d2[0][4], 'logs/gan_v2_diag2.txt D_diag run_step 1', 3)
    F.add('DiagTwoRealMeanMin', min(x[3] for x in d2), 'logs/gan_v2_diag2.txt D_diag lines', 3)
    F.add('DiagTwoRealMeanMax', max(x[3] for x in d2), 'logs/gan_v2_diag2.txt D_diag lines', 3)
    F.add('DiagTwoFakeMeanAbsMax', max(abs(x[4]) for x in d2), 'logs/gan_v2_diag2.txt D_diag lines', 3)
    F.add('DiagTwoNDiag', len(d2), 'logs/gan_v2_diag2.txt D_diag lines')
    F.add('DiagTwoLastDiagStep', d2[-1][0], 'logs/gan_v2_diag2.txt D_diag lines')
    F.add('DiagRealSmoothTarget', 0.9, 'label_smooth 0.9', 1)
    d3 = logs['DiagThree']['diag']
    F.add('DiagThreeStepOneReal', d3[0][3], 'logs/gan_v2_diag3.txt D_diag run_step 1', 3)
    F.add('DiagThreeStepOneFake', d3[0][4], 'logs/gan_v2_diag3.txt D_diag run_step 1', 3)
    d1 = logs['Diag']
    F.add('DiagMaxD', max(s['d'] for s in d1['steps'].values()), 'logs/gan_v2_diag.txt step lines', 4)
    F.add('DiagTwoMaxD', max(s['d'] for s in logs['DiagTwo']['steps'].values()), 'logs/gan_v2_diag2.txt', 4)
    F.add('RetrainMaxD', max(s['d'] for s in rr['steps'].values()), 'logs/gan_v2_retrain.txt', 4)
    # SPADE loss composition at the end of training (last epoch, step lines)
    for t in ('Spade', 'SpadeAerial'):
        lg = logs[t]
        last_ep = max(s['epoch'] for s in lg['steps'].values())
        lst = [s for s in lg['steps'].values() if s['epoch'] == last_ep]
        F.add(f'Log{t}LastAdv', sum(s['adv'] for s in lst) / len(lst), lg['file'], 2)
        F.add(f'Log{t}LastLOne', sum(s['l1'] for s in lst) / len(lst), lg['file'], 2)
        F.add(f'Log{t}LastPerc', sum(s['perc'] for s in lst) / len(lst), lg['file'], 1)

    # ------------------------------------------------------------ compute
    s1h = 0.0
    for t, r in s1.items():
        h = r['train_minutes'] / 60
        s1h += h
        F.add(f'GpuH{t}', h, f'{STAGE1_RUNS[t]}/results.json train_minutes', 2)
    F.add('GpuHSOneTotal', s1h, 'derived: sum Stage 1 runs', 1)
    s2h = 0.0
    for t, lg in logs.items():
        h = sum(e['min'] for e in lg['epochs'].values()) / 60
        s2h += h
        F.add(f'GpuH{t}', h, f'logs/{lg["file"]} sum of epoch times', 2)
    F.add('GpuHSTwoTotal', s2h, 'derived: sum of logged Stage 2 epoch times', 1)
    evh = sum(met[k]['elapsed_sec'] for k in CONFIG_KEYS) / 3600
    F.add('GpuHEval', evh, 'derived: sum elapsed_sec over A-E (excludes clean-FID/KID)', 2)
    F.add('GpuHRelabel', rl['minutes'] / 60, 'results_table.phase1.relabel.minutes', 2)
    F.add('GpuHTotalLogged', s1h + s2h + evh + rl['minutes'] / 60, 'derived: Stage1 + Stage2 + eval + relabel', 1)
    F.add('GpuHTotalTrainEval', s1h + s2h + evh, 'derived: Stage1 + Stage2 + eval', 1)
    F.add('SpadeAerialUntimedEpochs', min(sa['epochs']) - 1, 'logs/spade_aerial_v2.txt')
    F.add('GpuHSpadePerEpoch', sum(e['min'] for e in sp['epochs'].values()) / len(sp['epochs']), 'logs/spade_v2.txt', 1)

    # ------------------------------------------------------------ published reference numbers
    for name, row in PUBLISHED.items():
        for mt, v in row.items():
            dp = {'ssim': 3, 'sd': 2, 'psnr': 3, 'fid': 2, 'kid': 3}[mt]
            F.add(f'Pub{name}{mt.capitalize()}', v, 'CrossViewDiff (arXiv 2408.14765) Table 1, CVUSA', dp, 1 if mt == 'psnr' else
                  (0 if mt == 'fid' else dp))
    fids = [r['fid'] for r in PUBLISHED.values()]
    F.add('PubFidMin', min(fids), 'derived: min published FID', 2, 0)
    F.add('PubFidMax', max(fids), 'derived: max published FID', 2, 0)
    F.add('CrossViewDiffTable', 1, 'CrossViewDiff Table 1')
    return F


def fmt_fact(v, dp):
    if isinstance(v, bool) or v is None or isinstance(v, (str, list)):
        return str(v)
    if dp is None:
        return str(v)
    s = f'{v:.{dp}f}'
    return '0' + s[2:] if s.startswith('-0') and float(s) == 0 else s


def cmd_facts(args):
    F = build_facts()
    if args.json:
        print(json.dumps({k: v[0] for k, v in F.d.items()}, indent=1, default=str))
        return
    w = max(len(k) for k in F.d)
    print(f'# Verified facts sheet ({len(F.d)} entries). value at display precision | full value | source')
    for k, (v, src, dp, mindp, pct) in F.d.items():
        full = repr(v) if isinstance(v, float) else str(v)
        print(f'{k:<{w}}  {fmt_fact(v, dp):>12}  {full:<22}  {src}')
    if F.warnings:
        print(f'\n# {len(F.warnings)} warning(s)')
        for x in F.warnings:
            print(' -', x)
    else:
        print('\n# no cross-check warnings')


# ============================================================================ bib

def latex_escape(s: str) -> str:
    rep = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_'}
    s = re.sub(r'\s+', ' ', s).strip()
    for a, b in rep.items():
        s = s.replace(a, b)
    accents = {'á': r"{\'a}", 'é': r"{\'e}", 'í': r"{\'i}", 'ó': r"{\'o}", 'ú': r"{\'u}", 'ń': r"{\'n}",
               'ü': r'{\"u}', 'ö': r'{\"o}', 'ä': r'{\"a}', 'ç': r'{\c{c}}', 'ł': r'{\l}', 'ş': r'{\c{s}}',
               'è': r'{\`e}', 'ğ': r'{\u{g}}', 'ı': r'{\i}', 'ñ': r'{\~n}', 'Ö': r'{\"O}', 'ø': r'{\o}',
               'å': r'{\aa}', 'ë': r'{\"e}', 'ï': r'{\"i}', 'ô': r'{\^o}', 'â': r'{\^a}', 'š': r'{\v{s}}',
               'č': r'{\v{c}}', 'ć': r"{\'c}", 'ž': r'{\v{z}}', 'ó': r"{\'o}"}
    for a, b in accents.items():
        s = s.replace(a, b)
    return s


def short_author(name: str) -> str:
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    given = [x for x in parts[:-1] if x]
    initials = '~'.join('-'.join(y[0] + '.' for y in g.split('-') if y) for g in given)
    return f'{initials}~{parts[-1]}'


def fetch_arxiv(ids: list[str]) -> dict:
    import urllib.request
    import xml.etree.ElementTree as ET
    url = 'http://export.arxiv.org/api/query?id_list=' + ','.join(ids) + f'&max_results={len(ids)}'
    raw = urllib.request.urlopen(url, timeout=60).read()
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(raw)
    out = {}
    for e in root.findall('a:entry', ns):
        aid = e.find('a:id', ns).text.rsplit('/', 1)[-1]
        base = re.sub(r'v\d+$', '', aid)
        out[base] = {
            'version': aid,
            'title': re.sub(r'\s+', ' ', e.find('a:title', ns).text).strip(),
            'authors': [a.find('a:name', ns).text.strip() for a in e.findall('a:author', ns)],
            'published': e.find('a:published', ns).text[:4],
            'summary': re.sub(r'\s+', ' ', e.find('a:summary', ns).text).strip(),
        }
    return out


def search_arxiv(title_words: str, author: str) -> list[dict]:
    """Title + author search, used only to locate the correct id when a given id does not match."""
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    q = ' AND '.join([f'ti:{w}' for w in title_words.split()] + [f'au:{author}'])
    url = 'http://export.arxiv.org/api/query?search_query=' + urllib.parse.quote(q) + '&max_results=5'
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    hits = []
    for e in root.findall('a:entry', ns):
        aid = re.sub(r'v\d+$', '', e.find('a:id', ns).text.rsplit('/', 1)[-1])
        hits.append({'id': aid, 'title': re.sub(r'\s+', ' ', e.find('a:title', ns).text).strip(),
                     'authors': [a.find('a:name', ns).text.strip() for a in e.findall('a:author', ns)],
                     'published': e.find('a:published', ns).text[:4]})
    return hits


# id given in the brief that failed the title check -> (title words, first-author surname) to search for
SEARCH_ON_MISMATCH = {'1803.03396': ('Cross-View Image Synthesis Conditional GANs', 'Regmi')}
BRIEF_IDS_REJECTED = {'1803.03831': 'Regmi & Borji, CVPR 2018 (cross-view image synthesis)'}


def verify_crossviewdiff_table() -> list[str]:
    """Download the CrossViewDiff PDF, extract text with pdftotext, compare Table 1 CVUSA columns."""
    import subprocess
    import tempfile
    import urllib.request
    msgs = []
    tmp = tempfile.mkdtemp()
    pdf, txt = os.path.join(tmp, 'cvd.pdf'), os.path.join(tmp, 'cvd.txt')
    urllib.request.urlretrieve('https://arxiv.org/pdf/2408.14765', pdf)
    subprocess.run(['pdftotext', '-layout', pdf, txt], check=True)
    with open(txt, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith('Table 1:'))
    for name, row in PUBLISHED.items():
        label = PUBLISHED_ROW_LABEL[name]
        hit = next((l for l in lines[start:start + 20] if l.strip().startswith(label)), None)
        if hit is None:
            msgs.append(f'MISMATCH {name}: row not found')
            continue
        nums = [float(x) for x in re.findall(r'(?<![\w(])\d+\.\d+', hit)][:5]
        exp = [row['ssim'], row['sd'], row['psnr'], row['fid'], row['kid']]
        ok = len(nums) == 5 and all(abs(a - b) < 1e-9 for a, b in zip(nums, exp))
        msgs.append(f'{"OK      " if ok else "MISMATCH"} {name}: pdf {nums} vs tools.py {exp}')
    return msgs


def bibitems(meta: dict) -> str:
    lines = []
    for aid, (key, _kw, venue, year) in BIB.items():
        m = meta.get(aid)
        if m is None:
            lines.append(f'\\bibitem{{{key}}} \\todo{{authors}}, \\todo{{title}}. {venue}, {year}. arXiv:{aid}.')
            continue
        auth = [latex_escape(short_author(a)) for a in m['authors']]
        if len(auth) > 6:
            a = ', '.join(auth[:3]) + ', et~al'
        elif len(auth) > 1:
            a = ', '.join(auth[:-1]) + ', and ' + auth[-1] if len(auth) > 2 else ' and '.join(auth)
        else:
            a = auth[0]
        title = latex_escape(m['title'])
        v = 'arXiv preprint' if venue == 'arXiv preprint' else f'In \\emph{{{venue}}}' if venue != 'IEEE TPAMI' \
            else '\\emph{IEEE Transactions on Pattern Analysis and Machine Intelligence}'
        lines.append(f'\\bibitem{{{key}}}\n{a}.\n\\newblock {title}.\n\\newblock {v}, {year}.'
                     f' \\href{{https://arxiv.org/abs/{aid}}}{{arXiv:{aid}}}.')
    return '\n\n'.join(lines)


def cmd_bib(args):
    meta = {}
    if not args.offline:
        try:
            meta = fetch_arxiv(list(BIB) + list(BRIEF_IDS_REJECTED))
        except Exception as e:  # network failure is reported, never guessed around
            print(f'[!] arXiv API failed: {e}', file=sys.stderr)
    print('% arXiv check (title words must appear in the fetched title):')
    for aid, why in BRIEF_IDS_REJECTED.items():
        m = meta.pop(aid, None)
        print(f'%   REJECTED {aid} (id given in the brief): fetched title "{m["title"] if m else "?"}"; expected {why}')
    for aid, (key, kw, venue, year) in BIB.items():
        m = meta.get(aid)
        if m is None:
            print(f'%   MISSING  {aid} {key}')
            continue
        ok = all(w.lower() in m['title'].lower() for w in kw.split())
        print(f'%   {"OK      " if ok else "MISMATCH"} {aid} {m["version"]:<14} {m["published"]}  {key:<24} '
              f'{m["title"][:70]}  [{m["authors"][0]} +{len(m["authors"]) - 1}]')
        if not ok:
            meta.pop(aid)
            if aid in SEARCH_ON_MISMATCH:
                words, au = SEARCH_ON_MISMATCH[aid]
                try:
                    for h in search_arxiv(words, au):
                        print(f'%      search hit {h["id"]} {h["published"]} {h["title"]} [{", ".join(h["authors"])}]')
                except Exception as e:
                    print(f'%      search failed: {e}')
    if not args.offline:
        try:
            for msg in verify_crossviewdiff_table():
                print('%   table:', msg)
        except Exception as e:
            print(f'%   table: could not verify ({e})')
    print()
    if args.abstracts:
        for aid, (key, *_r) in BIB.items():
            if aid in meta:
                print(f'% [{key}] {meta[aid]["summary"]}\n')
        return
    print(bibitems(meta))


# ============================================================================ figures

def _lm_font():
    """Typeset plot text with LaTeX + lmodern (Type 1 fonts identical to the paper); fall back to the OTF."""
    import shutil
    import matplotlib
    from matplotlib import font_manager
    matplotlib.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                                'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
                                'axes.linewidth': 0.6})
    if shutil.which('latex') and not os.environ.get('A2G_NO_USETEX'):
        matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif',
                                    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}'})
        return
    if os.path.isfile(LM_FONT):
        font_manager.fontManager.addfont(LM_FONT)
        matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=LM_FONT).get_name()
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'mathtext.fontset': 'cm'})


def _save_jpg(img, path, max_w=1600, q=85):
    from PIL import Image
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    img.save(path, quality=q, optimize=True, progressive=True)
    kb = os.path.getsize(path) / 1024
    print(f'    wrote {os.path.relpath(path, ROOT)}  {img.width}x{img.height}  {kb:.0f} KB')
    if kb > 400:
        _save_jpg(img, path, max_w, q - 10)


def colorize(lbl):
    import numpy as np
    pal = np.array(LAYOUT_RGB, dtype=np.uint8)
    return pal[np.clip(lbl, 0, 3)]


def recolor_repo_palette(arr):
    """Map an RGB crop in the repo's red/green/blue/yellow palette to the Okabe-Ito layout colours."""
    import numpy as np
    src = np.array(REPO_RGB, dtype=np.float32)
    d = ((arr[..., None, :].astype(np.float32) - src[None, None]) ** 2).sum(-1)
    return colorize(d.argmin(-1))


def val_rows():
    with open(p('cvpr_val_v2.csv'), encoding='utf-8') as f:
        return [[x.strip() for x in line.split(',')] for line in f if line.strip()]


def make_row(cells, cell_w, cell_h, gap=8, square_first=False, bg=(255, 255, 255)):
    """Concatenate PIL images into one row of equal cells; images are resized to fill each cell."""
    from PIL import Image
    w = len(cells) * cell_w + (len(cells) - 1) * gap
    row = Image.new('RGB', (w, cell_h), bg)
    for j, im in enumerate(cells):
        if im is None:
            continue
        x0 = j * (cell_w + gap)
        if im.width == im.height:  # aerial tiles: square, centred in the cell
            s = cell_h
            row.paste(im.resize((s, s), Image.LANCZOS), (x0 + (cell_w - s) // 2, 0))
        else:
            row.paste(im.resize((cell_w, cell_h), Image.LANCZOS), (x0, 0))
    return row


def regen_configs(indices: list[int]) -> dict:
    """Re-run Stage 1 + the A, B, D, E generators of final_eval on a few test indices; verify LPIPS vs npz."""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    from PIL import Image
    cwd = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    try:
        from aerial_warp import PanoAerialWarper
        from eval_gan_pix2pix import load_generator
        from final_eval import CONFIG_ORDER, DEFAULT_FEAT_VAL, DEFAULT_POLAR_CKPT, load_polar_model
        from gan_layout_codec import class_indices_to_onehot, load_gt_label_tensor
        from gan_metrics import per_image_lpips, tensor_to_uint8_nhwc
        from train_gan_pix2pix import generator_forward
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rows = val_rows()
        polar = load_polar_model(DEFAULT_POLAR_CKPT, DEFAULT_FEAT_VAL, dev)
        feats = np.load(DEFAULT_FEAT_VAL, mmap_mode='r')
        warper = PanoAerialWarper((256, 512))
        out = {i: {} for i in indices}
        with torch.inference_mode():
            for i in indices:
                f = torch.from_numpy(np.array(feats[i:i + 1], dtype=np.float32)).to(dev)
                lg = Fnn.interpolate(polar(f).float(), size=(256, 512), mode='bilinear', align_corners=True)
                out[i]['pred'] = lg.argmax(1)[0].cpu().numpy().astype(np.uint8)
                gt_t = load_gt_label_tensor(rows[i][2], 4, 256, 512)
                out[i]['gt'] = ((gt_t + 1) / 2).argmax(0).numpy().astype(np.uint8)
                g = Image.open(rows[i][1]).convert('RGB').resize((512, 256), Image.BILINEAR)
                out[i]['real_t'] = torch.from_numpy((np.array(g, dtype=np.float32) / 127.5 - 1).transpose(2, 0, 1))
                out[i]['real'] = g
                out[i]['aerial'] = Image.open(rows[i][0]).convert('RGB')
                out[i]['warp_t'] = warper.warp_aerial_to_pano(rows[i][0])
            for cfg in CONFIG_ORDER:
                G, step = load_generator(cfg.gan_ckpt, 4, 64, dev, generator=cfg.generator, img_h=256, img_w=512,
                                         aux_aerial=cfg.aux_aerial)
                G.eval()
                for i in indices:
                    lab = out[i]['pred'] if cfg.layout == 'pred' else out[i]['gt']
                    lt = class_indices_to_onehot(lab, 4).unsqueeze(0).to(dev)
                    if cfg.layout == 'gt':
                        lt = load_gt_label_tensor(rows[i][2], 4, 256, 512).unsqueeze(0).to(dev)
                    aer = out[i]['warp_t'].unsqueeze(0).to(dev) if cfg.aux_aerial else None
                    fake = generator_forward(G, lt, aer, cfg.aux_aerial).float().cpu()
                    out[i][cfg.key] = Image.fromarray(tensor_to_uint8_nhwc(fake)[0])
                    out[i][f'lpips_{cfg.key}'] = float(per_image_lpips(out[i]['real_t'].unsqueeze(0), fake, dev)[0])
                del G
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
        warp_np = None
        for i in indices:
            w = out[i]['warp_t'].numpy().transpose(1, 2, 0)
            warp_np = ((w * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
            out[i]['warp'] = Image.fromarray(warp_np)
        return out
    finally:
        os.chdir(cwd)


def check_regen(out: dict, fe: str) -> float:
    import numpy as np
    worst = 0.0
    for k in ('A', 'B', 'C', 'D', 'E'):
        z = np.load(p(fe, f'{CONFIG_SLUGS[k]}_per_image.npz'))
        pos = {int(i): j for j, i in enumerate(z['index'])}
        for i, d in out.items():
            if f'lpips_{k}' in d:
                diff = abs(d[f'lpips_{k}'] - float(z['lpips'][pos[i]]))
                worst = max(worst, diff)
                print(f'    regen check idx {i:5d} cfg {k}: LPIPS {d[f"lpips_{k}"]:.4f} vs npz '
                      f'{float(z["lpips"][pos[i]]):.4f}  (|diff| {diff:.1e})')
    return worst


def fig_f1_strip():
    """F1 example strip from the repository's pipeline.py smoke test (test index 500)."""
    import numpy as np
    from PIL import Image
    d = p('outputs', 'dataset_pipeline_test', 'idx_00500')
    aerial = Image.open(p('test_pipeline_inputs', 'aerial_00500.jpg')).convert('RGB')
    lay = recolor_repo_palette(np.asarray(Image.open(os.path.join(d, 'layout.png')).convert('RGB')))
    warp = Image.open(os.path.join(d, 'warped_aerial.png')).convert('RGB')
    gen = Image.open(os.path.join(d, 'panorama.png')).convert('RGB')
    real = Image.open(p('test_pipeline_inputs', 'ground_00500.jpg')).convert('RGB').resize((512, 256))
    for name, im in (('aerial', aerial.resize((256, 256), Image.LANCZOS)), ('layout', Image.fromarray(lay)),
                     ('warp', warp), ('gen', gen), ('real', real)):
        _save_jpg(im, os.path.join(FIG, f'f1_{name}.jpg'), max_w=800, q=88)


def fig_f3():
    """Stage 1 qualitative comparison on the first three test pairs (existing showcase dumps)."""
    import numpy as np
    from PIL import Image
    rows = val_rows()
    idxs = [0, 1, 2]
    cw, ch = 528, 96

    def crop_pred_g(path):
        a = np.asarray(Image.open(path).convert('RGB'))
        return Image.fromarray(recolor_repo_palette(a[20:132, 1356:1916]))  # PRED-G panel

    def crop_polar(i):
        a = np.asarray(Image.open(p('experiments', 'phase2', 'runs', 'polar_full', 'showcase',
                                    f'test_{i:04d}.jpg')).convert('RGB'))
        return Image.fromarray(recolor_repo_palette(a[448:672, 224:1456]))

    layers = OrderedDict()
    layers['aerial'] = [Image.open(rows[i][0]).convert('RGB') for i in idxs]
    layers['real'] = [Image.open(rows[i][1]).convert('RGB') for i in idxs]
    layers['gt'] = [Image.fromarray(colorize(np.asarray(Image.open(rows[i][2])))) for i in idxs]
    layers['crossnet'] = [crop_pred_g(p('experiments', 'baseline', 'results', 'eval_v2_labels_full8884',
                                        f'val_{i:07d}.jpg')) for i in idxs]
    layers['phase1'] = [crop_pred_g(p('experiments', 'phase1', 'results', 'eval_v2_labels_full8884',
                                      f'val_{i:07d}.jpg')) for i in idxs]
    layers['ours'] = [crop_polar(i) for i in idxs]
    for name, cells in layers.items():
        h = 150 if name == 'aerial' else ch
        _save_jpg(make_row(cells, cw, h), os.path.join(FIG, f'f3_{name}.jpg'), q=88)


def fig_f4_and_failures(regen: bool):
    from PIL import Image
    F = build_facts()
    fe = F['FinalEvalDir']
    sel = [F[f'SelIdx{k}'] for k in ('Best', 'QOne', 'Median', 'QThree', 'Worst')]
    print(f'    F4 indices {sel}')
    if not regen:
        raise SystemExit('F4 needs regeneration (saved sample PNGs do not cover the quantile rows)')
    out = regen_configs(sel)
    worst = check_regen(out, fe)
    print(f'    max |LPIPS regen - npz| = {worst:.2e}')
    cw, ch = 312, 156
    for name in ('aerial', 'pred', 'gt', 'A', 'B', 'D', 'E', 'real'):
        cells = []
        for i in sel:
            d = out[i]
            if name in ('pred', 'gt'):
                cells.append(Image.fromarray(colorize(d[name])))
            else:
                cells.append(d[name])
        _save_jpg(make_row(cells, cw, ch), os.path.join(FIG, f'f4_{name}.jpg'), q=87)
    return worst


def fig_app_uncurated(regen: bool = True):
    """Appendix Figure 7: six uncurated Config E samples (fixed seed), same row layout as F4."""
    import numpy as np
    from PIL import Image
    F = build_facts()
    fe = F['FinalEvalDir']
    idx = np.random.RandomState(0).choice(8884, 6, replace=False).astype(int).tolist()
    print(f'    uncurated indices {idx}')
    if not regen:
        raise SystemExit('uncurated appendix figure needs regeneration')
    out = regen_configs(idx)
    z = np.load(p(fe, f'{CONFIG_SLUGS["E"]}_per_image.npz'))
    pos = {int(i): j for j, i in enumerate(z['index'])}
    lp = []
    worst = 0.0
    for i in idx:
        got = float(out[i]['lpips_E'])
        ref = float(z['lpips'][pos[i]])
        diff = abs(got - ref)
        worst = max(worst, diff)
        lp.append(got)
        print(f'    idx {i:5d}: LPIPS {got:.4f} vs npz {ref:.4f}  (|diff| {diff:.1e})')
    print(f'    max |LPIPS regen - npz| = {worst:.2e}')
    meta = os.path.join(FIG, 'app_uncurated_meta.txt')
    with open(meta, 'w', encoding='utf-8') as fh:
        fh.write('indices ' + ' '.join(str(i) for i in idx) + '\n')
        fh.write('lpips ' + ' '.join(f'{x:.4f}' for x in lp) + '\n')
    cw, ch = 312, 156
    for name in ('aerial', 'pred', 'E', 'real'):
        cells = []
        for i in idx:
            d = out[i]
            if name == 'pred':
                cells.append(Image.fromarray(colorize(d['pred'])))
            elif name == 'E':
                cells.append(d['E'])
            else:
                cells.append(d[name])
        _save_jpg(make_row(cells, cw, ch), os.path.join(FIG, f'app_fail_{name}.jpg'), q=87)
    return idx, lp, worst


def fig_warp():
    from PIL import Image
    for idx in (1269, 3807):
        src = p('experiments', 'stage2_aerial', 'warp_check',
                next(f for f in os.listdir(p('experiments', 'stage2_aerial', 'warp_check')) if f.endswith(f'idx{idx}.jpg')))
        _save_jpg(Image.open(src).convert('RGB'), os.path.join(FIG, f'app_warp_{idx}.jpg'), q=85)


def _save_pdf(fig, name):
    """Save a matplotlib figure; with usetex the Type 1 fonts are embedded whole, so subset them via Ghostscript."""
    import shutil
    import subprocess
    import matplotlib.pyplot as plt
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    gs = shutil.which('mgs') or shutil.which('gswin64c') or shutil.which('gs')
    if gs:
        tmp = path + '.tmp.pdf'
        r = subprocess.run([gs, '-q', '-dBATCH', '-dNOPAUSE', '-dSAFER', '-sDEVICE=pdfwrite',
                            '-dCompatibilityLevel=1.5', '-dSubsetFonts=true', '-dEmbedAllFonts=true',
                            '-dAutoRotatePages=/None', f'-sOutputFile={tmp}', path], capture_output=True)
        if r.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
        else:
            print(f'[!] ghostscript subsetting failed for {name}: {r.stderr[-300:]}', file=sys.stderr)
    print(f'    wrote {os.path.relpath(path, ROOT)}  {os.path.getsize(path) / 1024:.0f} KB')


def fig_f5():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _lm_font()
    F = build_facts()
    order = ['A', 'B', 'D', 'C', 'E']
    col = {'A': OKABE_ITO['blue'], 'B': OKABE_ITO['blue'], 'D': OKABE_ITO['blue'],
           'C': OKABE_ITO['vermillion'], 'E': OKABE_ITO['vermillion']}
    import matplotlib.ticker
    aerial = {'A': False, 'B': False, 'C': False, 'D': True, 'E': True}
    marker = {'A': 'D', 'B': 'o', 'C': 'o', 'D': 's', 'E': 's'}
    xs = [0, 1, 2, 3.4, 4.4]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.0))
    panels = [('Lpips', 'LPIPS $\\downarrow$', True, '{:.2f}'), ('Psnr', 'PSNR (dB) $\\uparrow$', True, '{:.0f}'),
              ('Fid', 'clean-FID $\\downarrow$', False, '{:.0f}')]
    for ax, (tok, lab, has_ci, fmt) in zip(axes, panels):
        vals = [F[f'Cfg{k}{tok}'] for k in order]
        for x, k, v in zip(xs, order, vals):
            if has_ci:
                lo, hi = F[f'Cfg{k}{tok}Lo'], F[f'Cfg{k}{tok}Hi']
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color='black', capsize=2.2, lw=0.8, capthick=0.8,
                            zorder=2)
            ax.plot(x, v, marker[k], ms=5.5, color=col[k], mfc=col[k] if aerial[k] else 'white', mew=1.3, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(order)
        ax.set_xlim(-0.6, 5.0)
        ax.set_title(lab, pad=3)
        lo_v, hi_v = min(vals), max(vals)
        span = hi_v - lo_v
        ax.set_ylim(lo_v - 0.18 * span, hi_v + 0.18 * span)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p, f=fmt: f.format(v)))
        ax.grid(axis='y', color='0.9', lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        if tok == 'Fid':
            yd, ye = F['CfgDFid'], F['CfgEFid']
            ax.plot([2.15, 4.4], [yd, yd], ls=':', color='0.4', lw=0.8, zorder=1)
            ax.annotate('', xy=(4.75, ye), xytext=(4.75, yd),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=0, shrinkB=0))
            ax.text(4.68, (yd + ye) / 2, f'oracle gap\n+{F["FidOracleGap"]:.1f}', fontsize=7, ha='right',
                    va='center')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls='', marker='o', color=OKABE_ITO['blue'], mfc='white', mew=1.3,
                      label='oracle layout'),
               Line2D([], [], ls='', marker='o', color=OKABE_ITO['vermillion'], mfc='white', mew=1.3,
                      label='predicted layout'),
               Line2D([], [], ls='', marker='s', color='0.35', label='filled: + warped aerial'),
               Line2D([], [], ls='', marker='D', color='0.35', mfc='white', mew=1.3, label='Pix2Pix U-Net (A)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
    _save_pdf(fig, 'f5_ladder.pdf')


def fig_f6():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    _lm_font()
    F = build_facts()
    styles = [(OKABE_ITO['vermillion'], '-', 1.0), (OKABE_ITO['black'], '--', 1.2),
              (OKABE_ITO['blue'], '-', 1.0), (OKABE_ITO['green'], '-', 1.0)]
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    floor = 5e-5
    ax.axhspan(F['HealthyDLo'], F['HealthyDHi'], color='0.92', zorder=0)
    ax.text(33.5, 0.16, "``healthy'' band in the training logs", fontsize=7, color='0.35', va='center', ha='right')
    for (tok, label), (c, ls, lw) in zip(F6_RUNS, styles):
        lg = parse_gan_log(GAN_LOGS[tok])
        st = lg['steps']
        start = (lg['resumed_g_step'] or 0)
        x = np.array([s - start for s in st]) / 1000.0
        y = np.maximum(np.array([v['d'] for v in st.values()]), floor)
        k = 5
        ys = np.exp(np.convolve(np.log(y), np.ones(k) / k, mode='same')) if len(y) > k else y
        if len(y) > k:
            ys[:k // 2], ys[-(k // 2):] = y[:k // 2], y[-(k // 2):]
        ax.plot(x, ys, color=c, ls=ls, lw=lw, label=label, zorder=3 if ls == '--' else 2)
    ax.set_yscale('log')
    ax.set_ylim(floor * 0.8, 1.0)
    ax.set_xlim(0, None)
    ax.set_xlabel('training step within the run (thousands)')
    ax.set_ylabel('D loss (LSGAN)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(loc='lower center', frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    _save_pdf(fig, 'f6_dloss.pdf')


def cmd_figures(args):
    os.makedirs(FIG, exist_ok=True)
    only = set(args.only.split(',')) if args.only else {'f1', 'f3', 'f4', 'f5', 'f6', 'app'}
    if 'f1' in only:
        print('[*] F1 example strip'); fig_f1_strip()
    if 'f3' in only:
        print('[*] F3 Stage 1 comparison'); fig_f3()
    if 'f4' in only:
        print('[*] F4 end-to-end grid'); fig_f4_and_failures(not args.no_regen)
    if 'uncurated' in only:
        print('[*] appendix uncurated samples (Figure 7)'); fig_app_uncurated(not args.no_regen)
    if 'app' in only:
        print('[*] appendix warp examples'); fig_warp()
    if 'f5' in only:
        print('[*] F5 ladder'); fig_f5()
    if 'f6' in only:
        print('[*] F6 discriminator dynamics'); fig_f6()


# ============================================================================ audit

NAME_PATTERNS = [  # identifiers that contain digits but are not results (removed before extraction)
    r'VGG-1[69]', r'VGG1[69]', r'DINOv2-S/14', r'ViT-S/14', r'SegFormer-B0', r'ADE20K', r'ADE-20K',
    r'RTX\s*(PRO\s*)?\d{4}', r'Quadro\s*P\d+', r'Inception-v3', r'relu\d\\?_\d', r'pool3', r'AlexNet',
    r'Stage[~\s]*[12]', r'Phase[~\s]*[12]', r'Table[~\s]*1 of', r'Okabe', r'S/14', r'Blackwell',
    r'4\s*GB', r'\bTPAMI\b',
]
FORBIDDEN = [r'state[- ]of[- ]the[- ]art', r'\bSOTA\b', r'outperform', r'photo-?realistic', r'\bnovel\b',
             r'dramatic', r'\bwe solve', r'\bstable GAN\b', r'\bstable (adversarial|game)']
REVIEW_WORDS = [r'\bsignifican', r'\bfirst\b', r'\bstable\b']


def _strip_braced(text, cmd):
    """Remove \\cmd{...} including nested braces (also optional [..] args)."""
    out, i = [], 0
    pat = re.compile(r'\\' + cmd + r'\*?(\[[^\]]*\])*\{')
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[j], 0)
            j += 1
        i = j
        out.append(' ')
    return ''.join(out)


def _unwrap(text, cmd):
    return re.sub(r'\\' + cmd + r'\{([^{}]*)\}', r'\1', text)


def tex_body(tex: str) -> tuple[str, str]:
    body = tex.split(r'\begin{document}', 1)[1].split(r'\end{document}', 1)[0]
    body = re.sub(r'(?<!\\)%.*', '', body)
    bib = ''
    if r'\begin{thebibliography}' in body:
        body, bib = body.split(r'\begin{thebibliography}', 1)
    return body, bib


def clean_for_numbers(body: str) -> str:
    t = re.sub(r'\\section\{ADE20K to four-class label map\}.*?(?=\\section\{Compute\})', ' ', body, flags=re.S)
    t = re.sub(r'\\begin\{(equation|align)\*?\}(.*?)\\end\{\1\*?\}', r'\\[\2\\]', t, flags=re.S)
    for cmd in ('label', 'ref', 'Cref', 'cref', 'eqref', 'cite', 'citep', 'citet', 'includegraphics', 'url',
                'texttt', 'setlength', 'hspace', 'vspace', 'rule', 'definecolor', 'textcolor', 'color',
                'colorbox', 'swatch', 'resizebox', 'scalebox', 'addlinespace', 'raisebox', 'sisetup',
                'begin', 'end', 'fontsize', 'renewcommand', 'arraystretch', 'phantom', 'pagestyle',
                'thispagestyle', 'captionsetup', 'linespread', 'newcolumntype', 'tikzset', 'usetikzlibrary',
                'makebox', 'parbox', 'cellwidth', 'rowimg', 'collabel', 'adjustbox'):
        t = _strip_braced(t, cmd) if cmd not in ('textcolor', 'color', 'makebox', 'parbox', 'raisebox',
                                                 'resizebox', 'scalebox', 'adjustbox') else \
            re.sub(r'\\' + cmd + r'(\[[^\]]*\])*\{[^{}]*\}' + (r'\{[^{}]*\}' if cmd == 'resizebox' else ''),
                   ' ', t)
    t = re.sub(r'\\href\{[^{}]*\}', ' ', t)
    t = re.sub(r'\\multicolumn\{\d+\}\{[^{}]*\}', ' ', t)
    t = re.sub(r'\\multirow\{\d+\}\{[^{}]*\}', ' ', t)
    t = re.sub(r'\\cmidrule(\([^)]*\))?\{[^{}]*\}', ' ', t)
    t = re.sub(r'\\\\\[[^\]]*\]', ' ', t)
    t = re.sub(r'\[[^\[\]]*\b(width|height|scale|trim|angle|font|anchor|inner sep|minimum|node distance|'
               r'xshift|yshift|line width|rounded corners|text width|align|draw|fill|right|left|above|below|'
               r'shift|opacity|thick|dashed|->|-latex|stealth|looseness|out|in|bend|label|pos|sep|frame)[^\[\]]*\]',
               ' ', t)
    # TikZ coordinates, numeric options and lengths
    t = re.sub(r'\\foreach[^{]*\{[^{}]*\}', ' ', t)
    t = re.sub(r'\b[a-zA-Z]+!\d+', ' ', t)  # xcolor mixes such as oiBlue!70
    t = re.sub(r'\(\s*-?[\d.]+\s*(cm)?\s*[,:]\s*-?[\d.]+\s*(cm)?\s*\)', ' ', t)
    t = re.sub(r'\(\s*\[[^\]]*\]\s*[\w.]+\s*\)', ' ', t)
    t = re.sub(r'\(\s*[a-zA-Z][\w.]*\s*\)', ' ', t)
    t = re.sub(r'-?\d*\.?\d+\s*(pt|em|ex|cm|mm|in|bp|sp)\b', ' ', t)
    t = re.sub(r'-?\d*\.?\d+\s*\\(linewidth|textwidth|columnwidth|textheight|tabcolsep|baselineskip|imgw|cellw)',
               ' ', t)
    for pat in NAME_PATTERNS:
        t = re.sub(pat, ' ', t)
    t = t.replace('{,}', ',').replace('−', '-').replace(r'\textminus', '-')
    t = re.sub(r'\$\s*-\s*\$\s*', '-', t)
    t = re.sub(r'\\num\{([^{}]*)\}', r' \1 ', t)
    t = re.sub(r'\\SI\{([^{}]*)\}\{[^{}]*\}', r' \1 ', t)
    t = re.sub(r'\\(times|pm|approx|sim|le|ge|,|;|!|quad|qquad)', ' ', t)
    t = re.sub(r'\\[a-zA-Z]+', ' ', t)  # remaining command names (arguments kept)
    return t


def math_spans(t: str) -> list[tuple[int, int]]:
    spans = [m.span() for m in re.finditer(r'\$[^$]*\$', t)]
    spans += [m.span() for m in re.finditer(r'\\\[.*?\\\]', t, re.S)]
    return spans


NUM_RE = re.compile(r'(?<![A-Za-z_\d.\\])(?:(?<=[\s$({\[=~/:;])|^)?([+-]?)(\d+(?:\.\d+)?(?:[eE]-?\d+)|'
                    r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)')


def extract_numbers(t: str):
    """Yield (string, value, decimals, has_sign, pos)."""
    for m in NUM_RE.finditer(t):
        sign, s = m.group(1), m.group(2)
        start = m.start(2)
        if start > 0 and t[start - 1] in '-' and start > 1 and t[start - 2] in '-':
            sign = ''  # en-dash range "24--41"
        if sign == '-' and start > 1 and (t[start - 2].isdigit() or t[start - 2] == '-'):
            sign = ''
        if start > 0 and t[start - 1].isalpha():
            continue
        try:
            v = float(s.replace(',', ''))
        except ValueError:
            continue
        dec = len(s.split('.')[1]) if '.' in s and 'e' not in s.lower() else 0
        if sign == '-':
            v = -v
        yield s, v, dec, bool(sign), m.start()


def whitelist(F: Facts):
    vals = []
    for name, (v, src, dp, mindp, pct) in F.d.items():
        items = v if isinstance(v, list) else [v]
        for x in items:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                continue
            if dp is None and float(x).is_integer():
                vals.append((name, float(x), 'int', 0))
            else:
                vals.append((name, float(x), 'float', mindp if mindp is not None else 0))
    # bibliography years / arXiv ids
    for aid, (_k, _kw, _v, yr) in BIB.items():
        vals.append(('bib', float(yr), 'int', 0))
    return vals


def match_number(v, dec, has_sign, wl):
    hits = []
    for name, x, kind, mindp in wl:
        for cand in ((x,) if has_sign else (x, -x)):
            if kind == 'int':
                if dec == 0 and abs(cand - v) < 1e-9:
                    hits.append(name)
                elif dec > 0 and abs(cand - v) < 1e-12:
                    hits.append(name)
            else:
                if dec < mindp:
                    continue
                if abs(cand - v) <= 0.5 * 10 ** (-dec) + 1e-9:
                    hits.append(name)
    return hits


def cmd_audit(args):
    F = build_facts()
    with open(TEX, encoding='utf-8') as f:
        tex = f.read()
    body, bib = tex_body(tex)
    problems = 0

    print('== 1. numbers in main.tex vs facts whitelist')
    wl = whitelist(F)
    t = clean_for_numbers(body)
    spans = math_spans(t)
    unmatched, n = [], 0
    for s, v, dec, sg, pos in extract_numbers(t):
        n += 1
        in_math = any(a <= pos < b for a, b in spans)
        if in_math and dec == 0 and abs(v) in (0, 1, 2):
            continue
        hits = match_number(v, dec, sg and not in_math, wl)  # in math a '-' is usually subtraction
        if not hits:
            ctx = t[max(0, pos - 50):pos + 30].replace('\n', ' ')
            unmatched.append((s, re.sub(r'\s+', ' ', ctx)))
    print(f'   {n} numbers checked, {len(unmatched)} unmatched')
    for s, ctx in unmatched:
        print(f'   UNMATCHED {s:>10}   ...{ctx}...')
    problems += len(unmatched)

    print('== 2. bibliography')
    keys = re.findall(r'\\bibitem\{([^}]+)\}', bib)
    cited = set()
    for m in re.finditer(r'\\cite[pt]?\{([^}]+)\}', body):
        cited |= {k.strip() for k in m.group(1).split(',')}
    known = {v[0] for v in BIB.values()}
    for k in keys:
        if k not in known:
            print(f'   bibitem {k} not in the tools.py BIB list'); problems += 1
        if k not in cited:
            print(f'   bibitem {k} never cited'); problems += 1
    for k in sorted(cited - set(keys)):
        print(f'   cited {k} has no bibitem'); problems += 1
    for aid in re.findall(r'arXiv:(\d{4}\.\d{4,5})', bib):
        if aid not in BIB:
            print(f'   arXiv id {aid} not in BIB list'); problems += 1
    bib_nums = [x for x in re.findall(r'(?<![\d.])(\d{4})(?![\d.])', re.sub(r'arXiv:\d{4}\.\d+|abs/\d{4}\.\d+', '', bib))]
    years = {str(v[3]) for v in BIB.values()}
    for y in bib_nums:
        if y not in years:
            print(f'   bibliography number {y} not a known venue year'); problems += 1
    print(f'   {len(keys)} bibitems, {len(cited)} cited keys')

    print('== 3. style rules')
    plain = re.sub(r'\\[a-zA-Z]+\*?', ' ', body)
    for pat in FORBIDDEN:
        for m in re.finditer(pat, plain, re.I):
            print(f'   FORBIDDEN "{m.group(0)}": ...{plain[max(0, m.start() - 60):m.end() + 40]}...'.replace('\n', ' '))
            problems += 1
    for pat in REVIEW_WORDS:
        for m in re.finditer(pat, plain, re.I):
            print(f'   review "{m.group(0)}": ...{re.sub(chr(10), " ", plain[max(0, m.start() - 70):m.end() + 50])}...')
    if re.search(r'\\(input|include)\{', body + tex.split(r'\begin{document}')[0]):
        print('   \\input/\\include found'); problems += 1
    ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, re.S).group(1)
    ab_plain = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', ab)
    ab_plain = re.sub(r'[{}$~\\]', ' ', ab_plain)
    nw = len([w for w in ab_plain.split() if re.search(r'[A-Za-z0-9]', w)])
    print(f'   abstract words: {nw} (target 170-200)')
    if not 170 <= nw <= 200:
        problems += 1
    todos = re.findall(r'\\todo\{((?:[^{}]|\{[^{}]*\})*)\}', body + bib)
    print(f'   {len(todos)} \\todo items:')
    for x in todos:
        print(f'     - {x}')
    nsub = len(re.findall(r'\\subsection\{', body.split(r'\section{Results}')[1].split(r'\section{')[0])) \
        if r'\section{Results}' in body else 0
    ntake = len(re.findall(r'\\textbf\{Takeaway\.\}', body.split(r'\section{Results}')[1].split(r'\section{')[0])) \
        if r'\section{Results}' in body else 0
    print(f'   Results subsections: {nsub}, Takeaway run-ins: {ntake}')

    print('== 4. assets and build')
    for fn in sorted(os.listdir(FIG)):
        kb = os.path.getsize(os.path.join(FIG, fn)) / 1024
        if kb > 400:
            print(f'   {fn} is {kb:.0f} KB (> 400 KB)'); problems += 1
        if fn.lower().endswith(('.jpg', '.png')):
            from PIL import Image
            w = Image.open(os.path.join(FIG, fn)).width
            if w > 1600:
                print(f'   {fn} is {w}px wide (> 1600)'); problems += 1
    used = set(re.findall(r'\\(?:includegraphics(?:\[[^\]]*\])?|rowimg)\{([^}]+)\}', body))
    used.discard('#1')
    for u in used:
        if not any(os.path.isfile(os.path.join(d, u)) for d in (PAPER, FIG)):
            print(f'   missing figure file {u}'); problems += 1
    pdf = os.path.join(PAPER, 'main.pdf')
    log = os.path.join(PAPER, 'main.log')
    if os.path.isfile(pdf):
        mb = os.path.getsize(pdf) / 2 ** 20
        print(f'   main.pdf {mb:.2f} MB')
        if mb > 5:
            problems += 1
    if os.path.isfile(log):
        lg = open(log, encoding='latin-1').read()
        for pat, what in ((r'LaTeX Warning: Reference .* undefined', 'undefined reference'),
                          (r'LaTeX Warning: Citation .* undefined', 'undefined citation'),
                          (r'^! ', 'error'), (r'There were undefined references', 'undefined refs (summary)'),
                          (r'Label\(s\) may have changed', 'rerun needed')):
            k = len(re.findall(pat, lg, re.M))
            if k:
                print(f'   log: {k} x {what}'); problems += k
        over = [float(x) for x in re.findall(r'Overfull \\[hv]box \(([\d.]+)pt too', lg)]
        big = [x for x in over if x > 5]
        print(f'   log: {len(over)} overfull boxes, {len(big)} wider than 5pt {big}')
        problems += len(big)
    aux = os.path.join(PAPER, 'main.aux')
    if os.path.isfile(aux):
        a = open(aux, encoding='latin-1').read()
        m = re.search(r'\\newlabel\{endofmain\}\{\{[^}]*\}\{(\d+)\}', a)
        if m:
            print(f'   main text ends on page {m.group(1)} (limit 10)')
            if int(m.group(1)) > 10:
                problems += 1
    print(f'\n== audit finished: {problems} problem(s)')
    if F.warnings:
        print('   fact cross-check warnings:', *F.warnings, sep='\n    - ')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('facts'); a.add_argument('--json', action='store_true')
    b = sub.add_parser('bib'); b.add_argument('--offline', action='store_true')
    b.add_argument('--abstracts', action='store_true', help='print arXiv abstracts (for writing related work)')
    c = sub.add_parser('figures'); c.add_argument('--no-regen', action='store_true'); c.add_argument('--only', default='')
    sub.add_parser('audit')
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding='utf-8')
    os.chdir(ROOT)
    if args.cmd == 'facts':
        cmd_facts(args)
    elif args.cmd == 'bib':
        cmd_bib(args)
    elif args.cmd == 'figures':
        cmd_figures(args)
    else:
        sys.exit(1 if cmd_audit(args) else 0)


if __name__ == '__main__':
    main()
