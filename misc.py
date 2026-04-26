import tensorflow.compat.v1 as tf
import pprint, os
import numpy as np
import imageio
from PIL import Image

pp = pprint.PrettyPrinter()

def pprint(obj):
  pp.pprint(obj)

def soft_labeling(im):
  im = im / (im.sum(-1)[..., None] + 1e-6)
  return im

def mkdir(dir_name):
  os.makedirs(dir_name, exist_ok=True)
  return dir_name

def mkdir_for_file(file_name):
  dir_name = os.path.dirname(file_name)
  mkdir(dir_name)
  return file_name

def basename(full_path):
  return os.path.basename(full_path)

# ---------------------------------------------------------------------------
# Image pre-processing (TF graph ops)
# ---------------------------------------------------------------------------

def preprocess_image(im, sz=None):
  """Normalise uint8 image tensor to [-1, 1] and optionally resize."""
  im = tf.cast(im, tf.float32) / 127.5 - 1.
  if sz:
    im = tf.image.resize(im, sz, method='bilinear')
  return im

def preprocess_label(label, num_class, sz):
  """Convert an int label map to a soft one-hot map downsampled to sz=(H, W)."""
  prob = tf.one_hot(label, num_class, axis=-1)
  prob = tf.cast(prob, tf.float32)
  prob = tf.image.resize(prob, sz, method='bilinear')
  prob = tf.reshape(prob, [-1, sz[0], sz[1], num_class])
  return prob

def proprocess_image(im_, old_range=(-1., 1.), new_range=(0., 1.)):
  """Remap a tensor from old_range to new_range for visualisation."""
  im = tf.cast(im_, tf.float32)
  im = (im - old_range[0]) / (old_range[1] - old_range[0])
  im = im * (new_range[1] - new_range[0]) + new_range[0]
  return im

# ---------------------------------------------------------------------------
# NumPy / PIL helpers
# ---------------------------------------------------------------------------

def center_crop(im, sz):
  """Centre-crop a numpy image to size [H, W]."""
  h, w = sz
  H, W = im.shape[:2]
  mx = int((W - w) / 2)
  my = int((H - h) / 2)
  return im[my:my + h, mx:mx + w, ...]

def _arr_to_uint8(arr):
  """Scale any float/int array to uint8 [0, 255]."""
  arr = np.asarray(arr, dtype=np.float32)
  lo, hi = arr.min(), arr.max()
  if hi > lo:
    arr = (arr - lo) / (hi - lo) * 255.
  else:
    arr = np.zeros_like(arr)
  return arr.clip(0, 255).astype(np.uint8)

def _pil_resize(arr, out_hw):
  """Resize a HxWxC (or HxW) array to out_hw=(H, W) using PIL bilinear."""
  h, w = out_hw
  arr_u8 = _arr_to_uint8(arr)
  if arr_u8.ndim == 2:
    arr_u8 = np.stack([arr_u8] * 3, axis=-1)
  return np.array(Image.fromarray(arr_u8).resize((w, h), Image.BILINEAR))

def to_montage(im_list, ncols=None, num_vis=6):
  """Assemble a list of [batch, H, W, C] numpy arrays into a single montage image."""
  if not ncols:
    batch = im_list[0].shape[0]
    ncols = min(batch, num_vis)
  nrows = len(im_list)

  max_dim = max(max(im.shape[1:3]) for im in im_list)
  h = w = min(max_dim, 256)
  image = np.zeros((h * nrows, w * ncols, 3), dtype=np.uint8)

  for i, im_batch in enumerate(im_list):
    for j, im in enumerate(im_batch[:ncols, ...]):
      r = i * h; c = j * w
      if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
      if im.shape[-1] > 3:
        im = im[..., [2, 3, 1]]   # (r,g,b) -> (road, tree, bldg)
      image[r:r + h, c:c + w, :] = _pil_resize(im, [h, w])

  return image

def pretty_transfmat(mat, source_size, target_size):
  hs, ws = source_size
  ht, wt = target_size
  mat_out = np.zeros([hs * ht, ws * wt])
  count = 0
  for i in range(ht):
    for j in range(wt):
      sr = i * hs; er = sr + hs
      sc = j * ws; ec = sc + ws
      mat_out[sr:er, sc:ec] = mat[:, count].reshape([hs, ws])
      count += 1
  return mat_out
