# CrossViewNet: Predicting Ground-Level Scene Layout from Aerial Imagery

A deep learning pipeline that takes an **aerial satellite image** as input and outputs a **synthesised ground-level panoramic view** — no ground-level camera required.

The system is a two-stage pipeline:

1. **CrossNet** (Stage 1) — a CNN trained with weak supervision that converts an aerial image into a semantic segmentation map (road, vegetation, building, sky).
2. **Pix2Pix GAN** (Stage 2) — a conditional image-to-image GAN that converts the semantic map into a photorealistic ground-level panorama.

This is a PyTorch reimplementation of the CVPR 2017 paper:  
> *Predicting Ground-Level Scene Layout from Aerial Imagery*, Zhai et al., CVPR 2017. [[PDF]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhai_Predicting_Ground-Level_Scene_CVPR_2017_paper.pdf)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
  - [Stage 1 — CrossNet (Semantic Prediction)](#stage-1--crossnet-semantic-prediction)
  - [Stage 2 — Pix2Pix GAN (Image Synthesis)](#stage-2--pix2pix-gan-image-synthesis)
- [Inference](#inference)
  - [Full Pipeline (Recommended)](#full-pipeline-recommended)
  - [Quick Random Demo](#quick-random-demo)
- [Outputs](#outputs)
- [Architecture Overview](#architecture-overview)
- [Results](#results)

---

## Project Structure

```
Aerial2Ground/
│
├── aerial_to_ground_final.py   # End-to-end inference: aerial image → ground panorama
├── infer_random_demo.py        # Quick demo: picks a random sample and runs inference
│
├── train_crossnet.py           # Stage 1 training script (CrossNet)
├── train_gan_pix2pix.py        # Stage 2 training script (Pix2Pix GAN)
│
├── crossnet_model.py           # CrossNet model definition (PyTorch)
├── gan_unet_model.py           # Pix2Pix Generator (U-Net) + Discriminator
├── crossnet_dataset.py         # Dataset loader (reads CSV file paths)
├── backbone_tf_legacy.py       # Legacy TF reference backbone (not used in training)
│
├── prepare_data.py             # Utility to prepare/verify dataset CSV files
├── cvpr_train.csv              # Training split (image paths)
├── cvpr_val.csv                # Validation split (image paths)
│
├── requirements.txt            # Python dependencies
├── aerial_to_ground_final.ipynb  # Notebook version of the final pipeline
│
├── data/                       # Place dataset images here (see Dataset Setup)
├── cvpr_subset/                # Small example subset for quick testing
└── outputs/
    ├── ckpts_pt/               # CrossNet checkpoints saved here during training
    ├── ckpts_gan/              # GAN checkpoints saved here during training
    ├── dump_pt/                # CrossNet training visualisations
    ├── dump_gan/               # GAN training montages
    └── pipeline_results/       # Saved inference output montages
```

---

## Requirements

**Python 3.9+** is recommended.

### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install Pillow numpy imageio scipy
```

Or install everything at once from the requirements file:

```bash
pip install -r requirements.txt
# Then install PyTorch separately with the correct CUDA version for your system:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **Windows GPU note:** TensorFlow 2.11+ does not support GPU on Windows. The project uses **PyTorch** for all training and inference. The `tf_legacy` files are kept as a reference implementation only and are not needed to run the project.

| Package | Version | Purpose |
|---|---|---|
| torch | >= 2.0.0 | Model training & inference |
| torchvision | >= 0.15.0 | VGG16 backbone (pretrained) |
| Pillow | >= 9.0.0 | Image loading |
| numpy | >= 1.23.0 | Array operations |
| imageio | >= 2.22.0 | Saving output images |
| scipy | >= 1.9.0 | Utilities |

---

## Dataset Setup

The model is trained on the **CVUSA dataset** — geo-tagged aerial/ground panorama pairs.

1. Download the full dataset from the [original Google Drive link](https://drive.google.com/open?id=0BzvmHzyo_zCAX3I4VG1mWnhmcGc).
2. Extract it so that the image files are accessible under the paths listed in `cvpr_train.csv` and `cvpr_val.csv`.
3. The CSV files contain one image path per line. Edit the paths in the CSV files (or in `crossnet_dataset.py`) to match your local directory layout if needed.

A small example subset is already included in `cvpr_subset/` — you can use this to verify the pipeline end-to-end without downloading the full dataset.

---

## Training

### Stage 1 — CrossNet (Semantic Prediction)

CrossNet takes a **224×224 aerial image** and predicts a **semantic segmentation map** with 4 classes (road, vegetation, building, sky/other).

```bash
# Default run — uses cvpr_train.csv, batch size 4, 10 epochs
python train_crossnet.py

# Custom options
python train_crossnet.py \
    --train_csv cvpr_train.csv \
    --val_csv   cvpr_val.csv \
    --batch_size 2 \
    --epochs 20 \
    --lr 1e-3 \
    --num_classes 4 \
    --ckpt_dir outputs/ckpts_pt \
    --dump_dir outputs/dump_pt

# Train unconditioned variant (no ground-view conditioning)
python train_crossnet.py --no_conditioned

# Disable Automatic Mixed Precision (if you get AMP errors)
python train_crossnet.py --no_amp

# Resume from a checkpoint
python train_crossnet.py --resume outputs/ckpts_pt/crossnet_step0050000.pt
```

**Key arguments for `train_crossnet.py`:**

| Argument | Default | Description |
|---|---|---|
| `--train_csv` | `cvpr_train.csv` | Path to training CSV |
| `--val_csv` | `cvpr_val.csv` | Path to validation CSV |
| `--batch_size` | `4` | Batch size |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-3` | Initial learning rate |
| `--lr_decay_steps` | `5000` | LR decay every N steps |
| `--lr_decay_rate` | `0.7` | Multiplicative LR decay factor |
| `--num_classes` | `4` | Number of semantic classes |
| `--no_pretrained` | off | Skip ImageNet weights for VGG16 |
| `--no_conditioned` | off | Use unconditioned transformation |
| `--no_amp` | off | Disable mixed precision (use FP32) |
| `--log_every` | `10` | Print loss every N steps |
| `--vis_every` | `100` | Save visualisation every N steps |
| `--save_every` | `500` | Save checkpoint every N steps |
| `--resume` | `''` | Path to `.pt` checkpoint to resume |

Checkpoints are saved to `outputs/ckpts_pt/crossnet_step<NNNNNNN>.pt`.

---

### Stage 2 — Pix2Pix GAN (Image Synthesis)

The GAN learns to convert semantic maps into realistic ground-level panoramas. **Train CrossNet first** before training the GAN.

```bash
# Full training (~35 k samples, several hours per epoch on a mid-range GPU)
python train_gan_pix2pix.py

# Quick demo run — 5 000 samples, 20 epochs (1–2 h total)
python train_gan_pix2pix.py --max_samples 5000 --epochs 20

# Resume from existing checkpoints
python train_gan_pix2pix.py \
    --resume_g outputs/ckpts_gan/G_step0010000.pt \
    --resume_d outputs/ckpts_gan/D_step0010000.pt
```

GAN checkpoints are saved to `outputs/ckpts_gan/` as `G_step<N>.pt` (generator) and `D_step<N>.pt` (discriminator). Only the latest 3 checkpoints are kept to save disk space.

---

## Inference

### Full Pipeline (Recommended)

`aerial_to_ground_final.py` runs both stages end-to-end on any aerial image and displays a 3-panel result:

```
[ Aerial Input ]  |  [ CrossNet Semantic Map ]  |  [ GAN Synthesised Ground View ]
```

```bash
# Basic usage — auto-detects latest checkpoints in outputs/
python aerial_to_ground_final.py --input path/to/aerial.jpg

# Specify checkpoints explicitly
python aerial_to_ground_final.py \
    --input        path/to/aerial.jpg \
    --crossnet_ckpt outputs/ckpts_pt/crossnet_step0088830.pt \
    --gan_ckpt      outputs/ckpts_gan/G_step0197766.pt

# Save result without displaying a window (useful on headless servers)
python aerial_to_ground_final.py --input path/to/aerial.jpg --no_display
```

The result montage is automatically saved to `outputs/pipeline_results/`.

---

### Quick Random Demo

Picks a random sample from `cvpr_train.csv` and runs inference using the latest available checkpoints:

```bash
python infer_random_demo.py
```

> If no GAN checkpoint exists, the script skips Stage 2 and only outputs the semantic map. Train the GAN first with `train_gan_pix2pix.py` to enable full synthesis.

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/ckpts_pt/` | CrossNet `.pt` checkpoints |
| `outputs/ckpts_gan/` | GAN Generator + Discriminator `.pt` checkpoints |
| `outputs/dump_pt/` | Visualisations saved during CrossNet training |
| `outputs/dump_gan/` | Montage JPEGs `[semantic | real ground | synthesised]` from GAN training |
| `outputs/pipeline_results/` | Final inference montages from `aerial_to_ground_final.py` |

---

## Architecture Overview

```
Aerial Image (224×224 RGB)
        │
        ▼
┌─────────────────────────────────┐
│  CrossNet (Stage 1)             │
│  ─ VGG16 aerial feature encoder │
│  ─ Conditioned transformation   │
│    network (view mapping)       │
│  ─ Segmentation decoder         │
└───────────────┬─────────────────┘
                │
                ▼
  Semantic Map (4 classes, upscaled to 256×512)
                │
                ▼
┌─────────────────────────────────┐
│  Pix2Pix GAN (Stage 2)         │
│  ─ U-Net Generator              │
│  ─ PatchGAN Discriminator       │
│  ─ L1 + Adversarial loss        │
└───────────────┬─────────────────┘
                │
                ▼
  Synthesised Ground Panorama (256×512 RGB)
```

**CrossNet** is trained with weak supervision: ground-level images are segmented by an off-the-shelf segmentation model (e.g., DeepLab) to produce pseudo-labels, and the model learns to predict those labels from the aerial view alone.

**The GAN** is trained on clean ground-truth semantic annotations so it learns a robust label → photo mapping. At inference, CrossNet's predicted maps are used as input instead.

---

## Results

After training, the pipeline predicts semantic classes including:

- Roads
- Vegetation / trees
- Buildings / structures
- Sky / other

Example results are saved during training to `outputs/dump_pt/` (CrossNet) and `outputs/dump_gan/` (GAN montages), so you can monitor quality as training progresses.

---

## Citation

```bibtex
@inproceedings{zhai2017predicting,
  title={Predicting Ground-Level Scene Layout from Aerial Imagery},
  author={Zhai, Menghua and Bessinger, Zachary and Workman, Scott and Jacobs, Nathan},
  booktitle={CVPR},
  year={2017}
}
```
