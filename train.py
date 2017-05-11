
import tensorflow as tf

from config import configuration
from model import basic_model, discriminate
from ops import inputs as inputs_ops

tf.logging.set_verbosity(tf.logging.INFO)

def gradient_compute(images, captions):
  """Compute gradient with policy gradient
  input: images: [batch_size, height, width, channel]
         captions: [batch_size, length]
  """
  return 0.5

def main(unusd_argv):
  model_config = configuration.ModelConfig()
  train_config = configuration.TrainingConfig()

  train_dir = "/home/mazm13/im2txtViaGAN/train"
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)
  else:
    tf.logging.info("Training directory: %s", train_dir)

  g = tf.Graph()
  with g.as_default():
    generator = basic_model.BasicModel(
      model_config, mode="train", train_inception=False)
    descriminator = discriminate.Discriminator(
      model_config, train_inception=False)

if __name__ == "__main__":
  tf.app.run()