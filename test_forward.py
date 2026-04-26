"""
test_forward.py
---------------
Sanity-check the full CrossNet forward pass on the 5 demo samples.

Run:
    python test_forward.py

Expected output:
    [*] model graph built.
    [*] loaded 5 samples from "data/data.csv"
    --- Forward pass ---
    aerial feature    : (8, 17, 17, 4)
    aerial2ground     : (8, 8, 40, 4)
    loss              : <finite float>
    [PASS] Forward pass completed successfully.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import misc, config, crossnet

DATA_DIR  = 'data/'
DATA_LIST = DATA_DIR + 'data.csv'

def main():
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    devs = [d.name for d in sess.list_devices()]
    gpus = [d for d in devs if 'GPU' in d.upper()]
    print('[*] Compute device:', gpus[0] if gpus else 'CPU')

    net = crossnet.CrossNet(sess)
    net.load_data(DATA_LIST, DATA_DIR)
    tf.global_variables_initializer().run()

    feed_dict = next(net.feed_dict_generator())

    feat_a, feat_a2g, prob_a, prob_a2g, loss_val = sess.run(
        [net.feat_aerial, net.feat_aerial2ground,
         net.prob_aerial, net.prob_aerial2ground,
         net.loss],
        feed_dict=feed_dict
    )

    print('\n--- Forward pass ---')
    print('aerial feature    :', feat_a.shape)
    print('aerial2ground     :', feat_a2g.shape)
    print('prob_aerial       :', prob_a.shape)
    print('prob_aerial2ground:', prob_a2g.shape)
    print('loss              :', loss_val)

    assert feat_a.shape[1:]  == (17, 17, 4), 'Bad aerial shape: %s'  % str(feat_a.shape)
    assert feat_a2g.shape[1:] == (8, 40, 4), 'Bad ground shape: %s'  % str(feat_a2g.shape)
    assert np.isfinite(loss_val),             'Loss is NaN/Inf: %s'   % loss_val

    print('\n[PASS] Forward pass completed successfully.')

if __name__ == '__main__':
  main()
