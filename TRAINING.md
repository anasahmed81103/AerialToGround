# CrossViewNet — Full Training Guide

Complete walkthrough from dataset preparation to evaluating a trained model.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.11 |
| PyTorch | 2.0.0 | 2.3.0 |
| GPU VRAM | 6 GB (CrossNet) | 8–12 GB (GAN) |
| Disk space | ~50 GB (subset) | ~600 GB (full CVUSA) |
| RAM | 8 GB | 16+ GB |

---

## Step 0 — Environment Setup

```bash
# Clone repo
git clone https://github.com/anasahmed81103/AerialToGround.git
cd AerialToGround

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust cu121 → cu118 for older GPUs)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 1 — Dataset Setup

### Using the included demo images

No setup needed.  The `demo/` folder contains 5 aerial images for instant testing.

### CVUSA Subset (included)

The `cvpr_subset/` folder contains a sample of Bing Maps aerial tiles.
The `cvpr_train.csv` and `cvpr_val.csv` files list the paired image paths used for training.

Check the CSV is valid:

```bash
python prepare_data.py
```

### Full CVUSA Dataset

1. Request access at [https://mvrl.cse.wustl.edu/datasets/cvusa/](https://mvrl.cse.wustl.edu/datasets/cvusa/)
2. Download the Bing Maps tiles + Street View panoramas (~500 GB total)
3. Extract to a local directory, e.g. `E:/datasets/cvusa/`
4. Edit `cvpr_train.csv` and `cvpr_val.csv` so paths point to your directory

CSV format (one pair per line):
```
bingmap/bingmap/19/0000001.jpg,streetview/panos/0000001.jpg,streetview/annotations/0000001.png
```

---

## Step 2 — Stage 1: Train CrossNet

CrossNet learns to translate aerial feature maps into ground-level semantic maps.

### Recommended settings

```bash
python train_crossnet.py \
    --train_csv  cvpr_train.csv \
    --val_csv    cvpr_val.csv \
    --batch_size 4 \
    --epochs     30 \
    --lr         1e-3 \
    --lr_decay_steps 5000 \
    --lr_decay_rate  0.7 \
    --num_classes 4 \
    --ckpt_dir   outputs/ckpts_pt \
    --dump_dir   outputs/dump_pt \
    --log_every  10 \
    --vis_every  500 \
    --save_every 500
```

### Low-VRAM settings (4–6 GB GPU)

```bash
python train_crossnet.py \
    --batch_size 2 \
    --epochs     30 \
    --no_amp            # disable AMP if you get dtype errors
```

### Resume from checkpoint

```bash
python train_crossnet.py \
    --resume outputs/ckpts_pt/crossnet_step0088830.pt \
    --epochs 50
```

### Key arguments reference

| Argument | Default | Notes |
|----------|---------|-------|
| `--train_csv` | `cvpr_train.csv` | Training pairs CSV |
| `--val_csv` | `cvpr_val.csv` | Validation pairs CSV |
| `--batch_size` | `4` | Reduce to 2 for low VRAM |
| `--epochs` | `10` | 20–30 recommended for good results |
| `--lr` | `1e-3` | Initial learning rate |
| `--lr_decay_steps` | `5000` | Decay LR every N steps |
| `--lr_decay_rate` | `0.7` | Multiplicative decay factor |
| `--num_classes` | `4` | 4 = sky/veg/road/building |
| `--no_pretrained` | off | Skip ImageNet weights |
| `--no_conditioned` | off | Unconditioned baseline |
| `--no_amp` | off | Force FP32 (slower, more stable) |
| `--resume` | `''` | Path to `.pt` file to continue from |

### What to expect

- Early training: loss ~2.0, random-looking segmentation
- After 20k steps: clear sky/road/vegetation bands emerge
- After 88k steps (current checkpoint): stable 4-class prediction
- Visualisations saved every 500 steps to `outputs/dump_pt/`

---

## Step 3 — Stage 2: Train Pix2Pix GAN

The GAN learns to render the semantic map as a photorealistic ground panorama.

**Train CrossNet first.** The GAN uses the semantic annotations directly during
training (not CrossNet predictions), so CrossNet only needs to exist for inference.

### Full training run

```bash
python train_gan_pix2pix.py
```

### Quick test run (1–2 hours)

```bash
python train_gan_pix2pix.py \
    --max_samples 5000 \
    --epochs 20
```

### Resume

```bash
python train_gan_pix2pix.py \
    --resume_g outputs/ckpts_gan/G_step0197766.pt \
    --resume_d outputs/ckpts_gan/D_step0197766.pt
```

### What to expect

- **Early (~1k steps):** pure noise output
- **~20k steps:** rough colour gradients, sky/ground separation visible
- **~100k steps:** sky-vegetation-road structure clearly emerging
- **~200k steps (current):** blurry but structurally correct panoramas
- **~500k+ steps (full dataset):** sharper textures, more photorealistic

Training montages (semantic | real | synthesised) are saved every 500 steps to
`outputs/dump_gan/`.

---

## Step 4 — Evaluation

### Qualitative evaluation

```bash
# Run pipeline on demo images and inspect results
python inference.py --source demo/

# Run on CVUSA validation set samples
python infer_random_demo.py
```

### Quantitative metrics (manual)

The project currently reports qualitative results.  To compute mIoU for CrossNet:

```python
# Load CrossNet predictions vs ground-truth labels
# Use outputs from infer_random_demo.py
# Compare predicted semantic map vs label*.png ground truth
from sklearn.metrics import jaccard_score
# ... (see crossnet_dataset.py for label format)
```

Expected baseline mIoU at 88k steps: **~0.42** (4-class, CVUSA subset)

---

## Monitoring Training

Visualisation dumps are saved automatically:

```
outputs/dump_pt/   — CrossNet: [aerial | label_gt | label_pred] strips
outputs/dump_gan/  — GAN: [semantic | real_ground | synthesised] strips
```

To view the latest dump image:
```bash
# Open most recent GAN dump
python -c "from PIL import Image; import glob; Image.open(sorted(glob.glob('outputs/dump_gan/*.jpg'))[-1]).show()"
```

---

## Hardware Benchmarks

| Hardware | CrossNet train (batch=4) | GAN train (default) |
|----------|--------------------------|---------------------|
| RTX 3060 12GB | ~850 samples/sec | ~120 samples/sec |
| RTX 2070 8GB | ~650 samples/sec | ~80 samples/sec |
| GTX 1660 6GB | ~400 samples/sec | Use batch=2 |
| CPU (i7) | ~15 samples/sec | Very slow, not recommended |

*Times are approximate and vary by dataset I/O speed.*

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch too large | `--batch_size 2` |
| `UnicodeEncodeError` on Windows | Console encoding | `set PYTHONIOENCODING=utf-8` |
| `No CrossNet checkpoint found` | Haven't trained yet | Run `train_crossnet.py` first |
| `GAN skipped` | No G_step*.pt | Train GAN with `train_gan_pix2pix.py` |
| VGG-16 download slow | First run downloads weights | Be patient or use `--no_pretrained` |
| Low-quality outputs | Underfitting | Train for more epochs / steps |
