# 📡 CrossViewNet: Predicting Ground-Level Scene Layout from Aerial Imagery

## 🚀 Overview

**CrossViewNet** is a deep learning-based system that predicts **ground-level semantic scene layouts** using only **aerial imagery**. Inspired by cross-view learning, the model bridges aerial and ground perspectives by learning a transformation between the two domains.

This approach reduces the need for expensive manual labeling by leveraging **weak supervision from ground-level images**.

---

## 🎯 Objectives

- Predict ground-level semantic segmentation from aerial images
- Learn cross-view feature transformation
- Enable applications such as:
  - 📍 Image geolocalization
  - 🧭 Orientation estimation
  - 🏙️ Scene synthesis

---

## 🧠 Key Contributions

- 🔄 Cross-view learning framework
- 🧩 Semantic label transfer without direct aerial annotation
- 🧮 Learned transformation between viewpoints
- 🌐 Multi-task capabilities (segmentation, localization, synthesis)

---

## 🏗️ System Architecture

The model consists of four main components:

1. **Aerial Feature Extractor**
   - CNN-based (e.g., VGG16 / ResNet)
   - Extracts features from aerial images

2. **Semantic Mapping Network**
   - Converts features into semantic representations

3. **Transformation Network**
   - Learns spatial mapping between aerial and ground views

4. **Ground-Level Prediction Module**
   - Outputs pixel-level semantic segmentation

---

## 🗂️ Dataset

- **CVUSA Dataset**
  - Geo-tagged aerial and ground image pairs
- Ground truth generated using pre-trained segmentation models

---

## ⚙️ Tech Stack

- Python
- TensorFlow / PyTorch
- OpenCV
- NumPy, Matplotlib

---

## 📊 Features

- ✅ Weakly supervised learning
- ✅ Cross-domain feature transfer
- ✅ Semantic segmentation
- ✅ Orientation & geolocation estimation
- ✅ Ground-view synthesis

---

## 🧪 Results

- Predicts semantic classes such as:
  - Roads
  - Vegetation
  - Buildings
- Improves aerial understanding without manual labels
- Generates approximate ground-level views

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Hamdan-Vohra/Aerial2Ground.git
cd Aerial2Ground
```

Crossnet, a cross-view supervised learning solution
========

This code is a tensorflow implementation of this paper, [Predicting Ground-Level Scene Layout from Aerial Imagery](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhai_Predicting_Ground-Level_Scene_CVPR_2017_paper.pdf).

And you are welcome to visit our [Project Page](http://cs.uky.edu/~ted/research/crossview/)

Dependencies
------------
* Tensorflow r1.1 or higher [Installation Page](https://www.tensorflow.org/versions/r1.1/install/).

Running
------------
Training:
```bash
$ python main.py
```

Deploying:
```bash
$ python main.py --is_training=False
```
Data
------------
This repo only contains some example data for training, the whole dataset can be found in [this link](https://drive.google.com/open?id=0BzvmHzyo_zCAX3I4VG1mWnhmcGc).
You would have to edit the dataset path in main.py in order to use it.

Note
------------
We plan to release evaluation code soon.
