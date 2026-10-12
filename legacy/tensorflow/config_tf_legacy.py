"""
config_tf_legacy.py  —  TensorFlow 1 CrossNet configuration flags (legacy)
===========================================================================
Defines tf.app.flags defaults (epochs, batch size, CSV paths, size containers).
Used by train_crossnet_tf_legacy.py and test_crossnet_tf_legacy.py.
"""
import tensorflow.compat.v1 as tf

flags = tf.app.flags
flags.DEFINE_integer("num_classes",     4,                "Number of semantic classes")
flags.DEFINE_integer("num_epochs",      10,               "Number of training epochs")
flags.DEFINE_integer("snapshot_iters",  1000,             "Save checkpoint every N steps")
flags.DEFINE_integer("batch_size",      8,                "Batch size")
flags.DEFINE_boolean("is_training",     True,             "True for training, False for deploy")
flags.DEFINE_boolean("batch_norm",      False,            "Use batch norm (needs tf_keras; disabled for TF 2.19 compat)")
flags.DEFINE_boolean("conditioned",     True,             "Condition transformation on aerial image content")
flags.DEFINE_string( "data_list",       "data/data.csv",  "CSV manifest: aerial,ground,label per line")
flags.DEFINE_string( "data_dir",        "",               "Optional path prefix for all entries in data_list")
default_config = flags.FLAGS

class SizeContainer:
  def __init__(self):
    self.H_src, self.W_src, self.C_src = 224, 224,  3    # aerial image (H, W, C)
    self.H_tar, self.W_tar, self.C_tar = 224, 1232, 3    # ground panorama (H, W, C)
    self.before_transf = [17, 17]   # aerial feature map after 4 VGG pooling stages
    self.after_transf  = [8,  40]   # ground feature map (target of transformation)
    self.image_aerial  = [self.H_src, self.W_src]
    self.image_ground  = [self.H_tar, self.W_tar]
