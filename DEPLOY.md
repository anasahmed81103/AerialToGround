# CrossViewNet — Deployment Guide

How to deploy CrossViewNet as a publicly accessible demo or production service.

---

## Option 1 — Hugging Face Spaces (Free, Recommended)

Hugging Face Spaces gives you a free GPU-accelerated web demo with a shareable URL.

### Requirements

- Hugging Face account (free): [https://huggingface.co/](https://huggingface.co/)
- Model checkpoints uploaded to Hugging Face Hub

### Step 1 — Upload checkpoints to HF Hub

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='outputs/ckpts_pt/crossnet_step0088830.pt',
    path_in_repo='crossnet_step0088830.pt',
    repo_id='YOUR_HF_USERNAME/crossviewnet-weights',
    repo_type='model',
)
api.upload_file(
    path_or_fileobj='outputs/ckpts_gan/G_step0197766.pt',
    path_in_repo='G_step0197766.pt',
    repo_id='YOUR_HF_USERNAME/crossviewnet-weights',
    repo_type='model',
)
"
```

### Step 2 — Create app.py (Gradio interface)

```python
# app.py  — drop this in the repo root for Hugging Face Spaces
import gradio as gr
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from crossnet_model import CrossNet
from gan_unet_model import UNetGenerator
import torch.nn.functional as F
import torchvision.transforms.functional as TF

NUM_CLASSES = 4
GAN_H, GAN_W = 256, 512
CLASS_COLORS = np.array([[255,64,64],[64,200,64],[64,64,255],[255,220,30]], dtype=np.uint8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download checkpoints from HF Hub on first run
def load_models():
    cn_path = hf_hub_download('YOUR_HF_USERNAME/crossviewnet-weights', 'crossnet_step0088830.pt')
    g_path  = hf_hub_download('YOUR_HF_USERNAME/crossviewnet-weights', 'G_step0197766.pt')
    
    model = CrossNet(num_classes=NUM_CLASSES, conditioned=True, pretrained=False).to(device)
    model.load_state_dict(torch.load(cn_path, map_location=device, weights_only=False)['model'])
    model.eval()
    
    G = UNetGenerator(in_ch=NUM_CLASSES, ngf=64).to(device)
    G.load_state_dict(torch.load(g_path, map_location=device, weights_only=False)['model'])
    G.eval()
    return model, G

crossnet, G = load_models()

def predict(aerial_pil: Image.Image):
    img = aerial_pil.convert('RGB').resize((224, 224), Image.BILINEAR)
    t = TF.normalize(TF.to_tensor(img), [0.5]*3, [0.5]*3).unsqueeze(0).to(device)
    with torch.no_grad():
        _, Lg = crossnet(t)
        prob_up = F.interpolate(torch.softmax(Lg,1), (224,1232), mode='bilinear', align_corners=True)
        pred = prob_up.argmax(1)[0].cpu().numpy()
        sem_pil = Image.fromarray(CLASS_COLORS[pred])
        
        B,C,H,W = Lg.shape
        one_hot = torch.zeros(B,C,H,W,device=device)
        one_hot.scatter_(1, Lg.argmax(1,keepdim=True), 1.0)
        one_hot = F.interpolate(one_hot, (GAN_H,GAN_W), mode='nearest')*2-1
        fake = G(one_hot)
        fake = F.interpolate(fake, (224,1232), mode='bilinear', align_corners=True)
        arr = ((fake[0].cpu().float().numpy()*0.5+0.5)*255).clip(0,255).astype(np.uint8).transpose(1,2,0)
        gan_pil = Image.fromarray(arr)
    return sem_pil, gan_pil

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Aerial Satellite Tile'),
    outputs=[
        gr.Image(type='pil', label='CrossNet Semantic Map'),
        gr.Image(type='pil', label='GAN Synthesised Ground View'),
    ],
    title='CrossViewNet: Aerial to Ground Synthesis',
    description='Upload a satellite/aerial tile to predict a ground-level panoramic view.',
    examples=[['demo/sample_aerial_road.jpg'], ['demo/sample_aerial_forest.jpg']],
)
demo.launch()
```

### Step 3 — Push to Spaces

1. Create a new Space at [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Hardware: **CPU Basic** (free) or **T4 Small** (GPU, ~$0.60/hr)
2. Add `app.py` and `requirements.txt` to the Space repo
3. The Space will build automatically

---

## Option 2 — Google Colab Demo

Create a shareable Colab notebook:

```python
# Cell 1 — Clone and install
!git clone https://github.com/anasahmed81103/AerialToGround.git
%cd AerialToGround
!pip install -r requirements.txt -q

# Cell 2 — Download checkpoints (from Google Drive or HF Hub)
# Option A: gdown from Drive
!pip install gdown -q
!gdown --id YOUR_DRIVE_FILE_ID -O outputs/ckpts_pt/crossnet_step0088830.pt
!gdown --id YOUR_DRIVE_FILE_ID -O outputs/ckpts_gan/G_step0197766.pt

# Cell 3 — Run inference
!python inference.py --source demo/ --no_display

# Cell 4 — Display results
from IPython.display import Image
Image('outputs/pipeline_results/sample_aerial_road_full_pipeline_montage.png')
```

---

## Option 3 — FastAPI Docker Container

For production or self-hosted deployment:

```python
# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io, torch, numpy as np
from PIL import Image

app = FastAPI(title='CrossViewNet API')

# Load models at startup...
# (same loading logic as inference.py)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    aerial = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # ... run inference ...
    # Return montage as JPEG
    buf = io.BytesIO()
    montage.save(buf, format='JPEG')
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/jpeg')
```

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt fastapi uvicorn python-multipart
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t crossviewnet .
docker run -p 8000:8000 --gpus all crossviewnet
# Visit http://localhost:8000/docs for the interactive API
```

---

## Option 4 — Gradio Local Demo (No Hosting)

For quick sharing on your local network:

```bash
pip install gradio
python app.py
# Gradio will print a local URL and an optional public share link
```

---

## Checkpoint Hosting

Current checkpoints (~250 MB total) are too large for a regular GitHub commit.
Recommended hosting options:

| Option | Storage | Speed | Cost |
|--------|---------|-------|------|
| Hugging Face Hub | Unlimited | Fast CDN | Free |
| Google Drive | 15 GB | Good | Free |
| GitHub Releases | 2 GB/release | GitHub CDN | Free |
| AWS S3 | Pay-per-use | Very fast | ~$0.023/GB/month |

### Upload to Hugging Face Hub

```bash
pip install huggingface_hub
huggingface-cli login
python -c "
from huggingface_hub import HfApi
api = HfApi()
# Create the repo first at huggingface.co
for f in ['outputs/ckpts_pt/crossnet_step0088830.pt',
          'outputs/ckpts_gan/G_step0197766.pt']:
    api.upload_file(path_or_fileobj=f, path_in_repo=f.split('/')[-1],
                    repo_id='YOUR_USERNAME/crossviewnet-weights', repo_type='model')
"
```

---

## Scaling Up: Training on Cloud

For training the full model (full CVUSA, 50+ epochs):

| Platform | GPU | Cost | Notes |
|----------|-----|------|-------|
| Google Colab Pro+ | A100 40GB | ~$50/month | 24h sessions |
| Vast.ai | RTX 3090 / A6000 | ~$0.30–0.80/hr | On-demand |
| RunPod | RTX 4090 | ~$0.74/hr | Reliable |
| Lambda Labs | A100 80GB | ~$1.99/hr | Dedicated |
| Kaggle | P100 / T4 | Free | 30 hrs/week |

### Kaggle Training Template

```bash
# On Kaggle notebook:
!git clone https://github.com/anasahmed81103/AerialToGround.git && cd AerialToGround
!pip install -r requirements.txt -q

# Upload dataset CSV via Kaggle Dataset
!python train_crossnet.py --train_csv /kaggle/input/cvusa-splits/cvpr_train.csv \
    --epochs 30 --batch_size 8

# Download checkpoint from output
```
