from utils import test_set, train_set
import tensorflow as tf
from mnist_capsnet import MNIST


flags = tf.app.flags

# environment
flags.DEFINE_string("env", "dev", "Depending on environment it will store additional data ['dev']")

# model params
flags.DEFINE_integer("epochs", 100, "Epoch to train [2]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam [0.001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [10]")

# data params
flags.DEFINE_string("tboard_dir", "./tensorboard_dir", "Tensorboard directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")


FLAGS = flags.FLAGS


if __name__ == '__main__':
    mnist = MNIST()
    mnist.fit()
