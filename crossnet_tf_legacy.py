"""
crossnet_tf_legacy.py  —  CrossNet class for TensorFlow 1 (legacy)
====================================================================
Original TF1 CrossNet graph, train/deploy loop, and checkpoint saving.
For the current PyTorch model, see crossnet_model.py.
"""
import tensorflow.compat.v1 as tf
import layers_tf_legacy as models, utils_tf_legacy as misc, config_tf_legacy as config, os, time
from random import shuffle
import numpy as np
import imageio
from PIL import Image

class CrossNet(object):
  def __init__(self, sess):
    self.sess        = sess
    self.szs         = config.SizeContainer()
    self.config      = config.default_config
    self.batch_size  = self.config.batch_size
    self.num_classes = self.config.num_classes
    self.conditioned = self.config.conditioned
    self.batch_norm  = self.config.batch_norm
    self.is_training = self.config.is_training

    self.log_dir  = misc.mkdir('outputs/logs/')
    self.ckpt_dir = misc.mkdir('outputs/ckpts/')
    self.dump_dir = misc.mkdir('outputs/dump/{}'.format('train' if self.is_training else 'deploy'))

    self.image_aerial_holder = tf.placeholder(tf.float32, [self.batch_size, None, None, self.szs.C_src])
    self.image_ground_holder = tf.placeholder(tf.float32, [self.batch_size, None, None, self.szs.C_tar])
    self.label_ground_holder = tf.placeholder(tf.int32,   [self.batch_size, None, None])

    self.build_model(
        [self.image_aerial_holder, self.image_ground_holder, self.label_ground_holder],
        self.is_training)

  # ------------------------------------------------------------------
  # Data loading
  # ------------------------------------------------------------------
  def load_data(self, image_list, image_dir=''):
    self.data_names = []
    with open(image_list, 'r') as fid:
      for line in fid.readlines():
        parts = [p.strip() for p in line.strip().split(',')]
        if len(parts) != 3:
          continue
        if image_dir:
          parts = [os.path.join(image_dir, p) for p in parts]
        self.data_names.append(dict(zip(['im_a', 'im_g', 'lb_g'], parts)))
    shuffle(self.data_names)
    self.num_samples = len(self.data_names)
    misc.pprint('[*] loaded %d samples from "%s"' % (self.num_samples, image_list))

  # ------------------------------------------------------------------
  # Image I/O helpers (scipy-free)
  # ------------------------------------------------------------------
  @staticmethod
  def _load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))

  @staticmethod
  def _load_label(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
      arr = arr[..., 0]
    return arr.astype(np.uint8)

  @staticmethod
  def _resize_rgb(arr, hw):
    H, W = hw
    return np.array(Image.fromarray(arr).resize((W, H), Image.BILINEAR))

  @staticmethod
  def _resize_label(arr, hw):
    H, W = hw
    return np.array(Image.fromarray(arr).resize((W, H), Image.NEAREST))

  # ------------------------------------------------------------------
  # Batch generator
  # ------------------------------------------------------------------
  def feed_dict_generator(self):
    for ib in range(0, self.num_samples, self.batch_size):
      batch_a, batch_g, batch_l = [], [], []
      for ix in range(self.batch_size):
        names = self.data_names[(ib + ix) % self.num_samples]
        im_a  = self._load_rgb(names['im_a'])
        im_g  = self._load_rgb(names['im_g'])
        lb_g  = self._load_label(names['lb_g'])
        im_a  = misc.center_crop(im_a, self.szs.image_aerial)
        im_g  = self._resize_rgb(im_g,   self.szs.image_ground)
        lb_g  = self._resize_label(lb_g, self.szs.image_ground)
        batch_a.append(im_a); batch_g.append(im_g); batch_l.append(lb_g)
      yield {
          self.image_aerial_holder: np.array(batch_a),
          self.image_ground_holder: np.array(batch_g),
          self.label_ground_holder: np.array(batch_l),
      }

  # ------------------------------------------------------------------
  # Graph construction
  # ------------------------------------------------------------------
  def build_model(self, data, is_training=True):
    raw_aerial, raw_ground, label_ground = data
    self.image_aerial = misc.preprocess_image(raw_aerial, self.szs.image_aerial)
    self.image_ground = misc.preprocess_image(raw_ground, self.szs.image_ground)
    self.prob_ground  = misc.preprocess_label(label_ground, self.num_classes, self.szs.after_transf)
    self.im_aerial    = misc.proprocess_image(self.image_aerial)
    self.im_ground    = misc.proprocess_image(self.image_ground)

    self.feat_aerial = models.pixelnet(
        self.image_aerial, self.num_classes,
        is_training=is_training, batch_norm=self.batch_norm)
    misc.pprint(self.feat_aerial.get_shape().as_list())

    feat_aerial_small = (self.feat_aerial if is_training
        else tf.image.resize(self.feat_aerial, self.szs.before_transf, method='bilinear'))

    weights = models.compute_transfweights(
        self.szs.before_transf, self.szs.after_transf,
        self.conditioned, is_training=is_training, batch_norm=self.batch_norm)
    self.feat_aerial2ground = models.transfnet(feat_aerial_small, weights, self.szs.after_transf)

    if is_training:
      self.merged     = tf.summary.merge_all()
      self.summarizer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

      with tf.name_scope('Loss'):
        self.loss_class = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.prob_ground, logits=self.feat_aerial2ground))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss_reg = tf.add_n(reg_losses) if reg_losses else tf.constant(0.0)
        self.loss = self.loss_class + self.loss_reg

      with tf.name_scope('Optimizer'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
          self.step  = tf.Variable(0, name='global_step', trainable=False)
          lr         = tf.compat.v1.train.exponential_decay(
              0.001, self.step, 5000, 0.7, staircase=True)
          self.optim = tf.compat.v1.train.AdamOptimizer(lr).minimize(
              self.loss, global_step=self.step)

    self.saver = tf.compat.v1.train.Saver(max_to_keep=10)
    misc.pprint('[*] model graph built.')

    self.transfweights, self.transfbiases = tf.get_collection('transformer_weights')
    self.prob_aerial        = tf.nn.softmax(self.feat_aerial,        axis=-1)
    self.prob_aerial2ground = tf.nn.softmax(self.feat_aerial2ground, axis=-1)

    with tf.name_scope('Vis'):
      self.visual = [
          self.im_aerial,
          self.im_ground,
          self.prob_aerial,
          self.prob_ground,
          self.prob_aerial2ground,
          self.transfweights,
      ]

  # ------------------------------------------------------------------
  # Checkpoint helpers
  # ------------------------------------------------------------------
  def restore(self):
    ckpt = tf.compat.v1.train.get_checkpoint_state(self.ckpt_dir)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    misc.pprint('[*] restored checkpoint from "%s".' % self.ckpt_dir)

  def save(self):
    self.saver.save(self.sess, '%s/model.ckpt' % self.ckpt_dir, global_step=self.step)

  # ------------------------------------------------------------------
  # Train / deploy loop
  # ------------------------------------------------------------------
  def train_test(self):
    if self.is_training:
      tf.global_variables_initializer().run()
      num_epochs = self.config.num_epochs
    else:
      self.restore()
      num_epochs = 1

    step = 0
    for iEpoch in range(num_epochs):
      for feed_dict in self.feed_dict_generator():
        if self.is_training:
          tic = time.time()
          _, summary, loss, step = self.sess.run(
              [self.optim, self.merged, self.loss, self.step], feed_dict)
          toc = time.time()
          print('[epoch %d] [step %06d] loss: %.5f  (%.3f s/step)'
              % (iEpoch, step, loss, toc - tic))
        else:
          step += 1
          print('[deploy] step: %d' % step)

        visual = self.sess.run(self.visual, feed_dict)

        if step % 100 == 1 or not self.is_training:
          montage   = misc.to_montage(visual)
          save_path = misc.mkdir_for_file('%s/%06d.jpg' % (self.dump_dir, step))
          imageio.imwrite(save_path, montage)

        if self.is_training:
          if step % 50 == 1:
            self.summarizer.add_summary(summary, step)
          if step % self.config.snapshot_iters == 0:
            self.save()

    if self.is_training:
      self.save()
