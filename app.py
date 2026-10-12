"""
Gradio demo — aerial tile -> layout + warp + panorama (config E).

  pip install -r requirements_demo.txt
  python app.py
"""

from __future__ import annotations

import json
import os

import gradio as gr
import numpy as np
from PIL import Image

from pipeline import AerialToGround, load_aerial_uint8, read_val_row, save_montage

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_CSV = os.path.join(ROOT, 'cvpr_val_v2.csv')
TILES_JSON = os.path.join(ROOT, 'demo', 'val_tiles.json')

NOTE = (
    'Works best on **CVUSA-style Bing zoom-19 nadir ~750×750 tiles**. '
    'Other scales or oblique views may degrade layout and synthesis.'
)


def _load_tiles():
    with open(TILES_JSON, encoding='utf-8') as f:
        return json.load(f)['tiles']


def _init_pipeline():
    return AerialToGround(device='auto')


PIPE = None


def get_pipe():
    global PIPE
    if PIPE is None:
        PIPE = _init_pipeline()
    return PIPE


def run_on_path(aerial_path: str, ground_path: str | None = None):
    pipe = get_pipe()
    out = pipe.predict(aerial_path)
    aerial_u8 = load_aerial_uint8(aerial_path)
    montage = None
    tmp = os.path.join(ROOT, 'demo', '_tmp_montage.png')
    save_montage(
        aerial_u8, out['layout_rgb'], out['warped_aerial_rgb'], out['panorama_rgb'], tmp,
    )
    montage = np.array(Image.open(tmp))
    gt = None
    if ground_path and os.path.isfile(ground_path):
        gt = np.array(Image.open(ground_path).convert('RGB').resize(
            (out['panorama_rgb'].shape[1], out['panorama_rgb'].shape[0]), Image.BILINEAR,
        ))
    return (
        aerial_u8,
        out['layout_rgb'],
        out['warped_aerial_rgb'],
        out['panorama_rgb'],
        gt,
        montage,
    )


def on_upload(image):
    if image is None:
        return (None,) * 6
    import tempfile
    path = os.path.join(tempfile.gettempdir(), 'a2g_upload.jpg')
    Image.fromarray(image).save(path)
    return run_on_path(path, None)


def on_val_tile(tile_label: str):
    tiles = _load_tiles()
    rec = next(t for t in tiles if t['label'] == tile_label)
    row = read_val_row(VAL_CSV, rec['idx'])
    ap = os.path.join(ROOT, row['aerial'])
    gp = os.path.join(ROOT, row['ground'])
    return run_on_path(ap, gp)


def build_ui():
    tiles = _load_tiles()
    labels = [t['label'] for t in tiles]
    with gr.Blocks(title='Aerial2Ground') as demo:
        gr.Markdown('# Aerial2Ground — aerial → layout → panorama')
        gr.Markdown(NOTE)
        with gr.Tab('Upload aerial'):
            up = gr.Image(label='Aerial tile', type='numpy')
            btn_u = gr.Button('Run pipeline')
        with gr.Tab('CVUSA val examples'):
            sel = gr.Dropdown(labels, value=labels[0], label='Val tile')
            btn_v = gr.Button('Run on val pair')
        with gr.Row():
            o_aerial = gr.Image(label='Aerial input')
            o_layout = gr.Image(label='Predicted layout')
            o_warp = gr.Image(label='Warped aerial')
            o_pano = gr.Image(label='Generated panorama')
        o_gt = gr.Image(label='Real panorama (val tiles only)')
        o_mont = gr.Image(label='Montage')
        btn_u.click(on_upload, up, [o_aerial, o_layout, o_warp, o_pano, o_gt, o_mont])
        btn_v.click(on_val_tile, sel, [o_aerial, o_layout, o_warp, o_pano, o_gt, o_mont])
    return demo


if __name__ == '__main__':
    build_ui().launch(server_name='0.0.0.0')
