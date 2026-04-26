# Aerial2Ground — Research Analysis & Implementation Plan  
**Paper:** *Predicting Ground-Level Scene Layout from Aerial Imagery*  
**Authors:** Zhai et al., CVPR 2017  
**arXiv:** 1612.02709  

---

## 📊 Overview
- **10** critiques identified  
- **8** research gaps  
- **3** days to implement  
- **4** feasible contributions  

---

## 📌 Paper Summary
CrossNet predicts **pixel-level ground-view semantic segmentation** using only a **co-located aerial image**.

- **Classes:** sky, building, road, vegetation  
- **Backbone:** VGG16  
- **Features:** Hypercolumn from conv1–conv4  
- **Transformation:** Learned linear matrix **M**, conditioned via MLP (**F**)  
- **Mapping:**  
  - Aerial: `17×17` feature map  
  - Ground: `8×40` label map  
- **Training:** Cross-entropy loss with **SegNet-generated pseudo-labels**  
- **Dataset:** CVUSA (35,532 train / 8,884 test)  

### Applications
1. Weakly-supervised aerial segmentation  
2. Pre-training for ISPRS  
3. Orientation estimation  
4. Panorama synthesis  

---

## ⚠️ Critiques (Ordered by Severity)

1. **Weak supervision:**  
   SegNet-generated labels are noisy and outdated.

2. **Limited semantic classes (4 only):**  
   Not sufficient for real-world deployment.

3. **Linear transformation (M):**  
   Cannot model complex perspective distortions.

4. **Outdated backbone (VGG16):**  
   Heavy (138M params), no skip connections.

5. **No spatial attention:**  
   Global conditioning ignores spatial relevance.

6. **Low spatial resolution:**  
   17×17 → 8×40 limits fine details.

7. **Strict alignment assumption:**  
   Requires north-aligned panoramas.

8. **Temporal mismatch:**  
   Aerial vs ground captured at different times.

9. **No uncertainty modeling:**  
   Overconfident predictions in occluded regions.

10. **Domain transfer issues:**  
    Requires manual preprocessing for new datasets.

---

## 🔍 Research Gaps & Opportunities

| Gap | Description | Feasibility | Impact |
|-----|------------|------------|--------|
| **Richer Supervision** | Replace SegNet with DeepLabV3+ / SegFormer | 2–3 days | High |
| **Modern Backbone** | Use ResNet50 / MobileNetV2 | 1–2 days | High |
| **Attention Mapping** | Add spatial/cross-attention to F | 2–3 days | High |
| **Multi-scale Features** | Add conv5 + FPN fusion | 1 day | Medium |
| **Data Augmentation** | Flip, jitter, rotation | 0.5 day | Medium |
| **Temporal Robustness** | Filter by seasonal similarity | 2–3 days | Medium |
| **More Classes (6+)** | Match ISPRS taxonomy | 1 day | Medium |
| **Better Orientation Metric** | Add mean angular error | 0.5 day | Low |

---

## 🛠️ Codebase Issues (Must Fix)

| File | Issue | Fix |
|------|------|-----|
| main.py | `__flags` private access | Use `vars()` |
| crossnet.py | `xrange` | Replace with `range()` |
| crossnet.py | Deprecated loss API | Use `logits=`, `labels=` |
| crossnet.py | Softmax axis missing | Add `axis=-1` |
| misc.py | `scipy.misc` removed | Use PIL / skimage |
| misc.py | `xrange` | Replace with `range()` |
| models.py | `dim=` argument | Replace with `axis=` |
| nets.py | `contextlib.nested` | Use `ExitStack` |

---

## 🚀 3-Day Implementation Plan

### ✅ Day 1 — Baseline Setup
- Port Python 2 → Python 3  
- Migrate TensorFlow → TF2 or PyTorch  
- Run forward pass on demo data  
- Verify pipeline: conv1–4 → 17×17 → transform → 8×40
- Enable TensorBoard logging  

---

### ⚙️ Day 2 — Core Improvements
- Replace VGG16 → **ResNet50 (ImageNet pretrained)**  
- Update hypercolumn: res2, res3, res4, res5
- Add **CBAM attention** in transformation network  
- Replace SegNet with **DeepLabV3+ labels**  
- Add augmentations:
- Horizontal flip  
- Color jitter  

---

### 📈 Day 3 — Evaluation & Results
- Train:
- Baseline (VGG16)
- Improved (ResNet50 + Attention)
- Evaluate:
- Per-class IoU  
- Mean IoU  
- Orientation error (mAE)  
- Visualizations: aerial | aerial seg | ground pred | GT

- Write report contributions  

---

## 🧠 Proposed Architecture Changes

### 🔴 Original
- Backbone: VGG16 (138M)  
- Hypercolumn: conv1–4  
- Transformation: Linear MLP  
- Supervision: SegNet  
- Augmentation: None  
- Framework: TF 1.1  

---

### 🟢 Improved
- Backbone: **ResNet50 (25M)**  
- Features: **FPN-style multi-scale fusion**  
- Transformation: **CBAM attention**  
- Supervision: **DeepLabV3+ labels**  
- Augmentation: Flip + jitter  
- Framework: **PyTorch 2.x**  

---

## 📊 Expected Results

| Configuration | Classes | Est. mIoU | Status |
|--------------|--------|----------|--------|
| VGG16 + SegNet | 4 | ~0.42 | Baseline |
| VGG16 + DeepLab | 4 | ~0.50+ | Contribution A |
| ResNet50 + SegNet | 4 | ~0.48+ | Contribution B |
| ResNet50 + Attention + DeepLab | 4 | **~0.55+** | Target |
| ResNet50 + FPN + 6-class | 6 | TBD | Stretch |

---

## 🧾 Contribution Statement

We reproduce the CrossNet baseline and introduce three improvements:

1. **ResNet50 backbone** for stronger aerial feature extraction  
2. **CBAM attention module** for spatially-aware cross-view mapping  
3. **DeepLabV3+ supervision** for cleaner training labels  

These changes directly address the most critical limitations of the original paper while remaining feasible within a **2–3 day implementation window**, yielding improved segmentation performance on the CVUSA dataset.

---

## 📅 Metadata
- Generated: **Apr 26, 2026**  
- Project: **Aerial2Ground**  
- Course: **Semester 8 – Computer Vision**