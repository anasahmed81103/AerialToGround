import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # suppress TF C++ INFO/WARNING noise

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

import misc, config, crossnet

def main(_):
  cfg = config.default_config

  # Report compute device before building the session
  _probe = tf.Session(config=tf.ConfigProto(log_device_placement=False))
  available = [d.name for d in _probe.list_devices()]
  _probe.close()
  gpus = [d for d in available if 'GPU' in d.upper()]
  if gpus:
    print(f"[*] Compute device : GPU  {gpus[0]}")
  else:
    print("[!] Compute device : CPU  (no GPU detected — see README for GPU setup on Windows)")

  gpu_options = tf.GPUOptions(allow_growth=True)
  cfg_proto   = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

  with tf.Session(config=cfg_proto) as sess:
    network = crossnet.CrossNet(sess)
    network.load_data(cfg.data_list, cfg.data_dir)
    misc.pprint(vars(cfg))
    network.train_test()

if __name__ == '__main__':
  tf.app.run()
