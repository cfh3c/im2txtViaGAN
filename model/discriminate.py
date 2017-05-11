
import tensorflow as tf

from ops import drm_image_embedding
from ops import drm_image_processing

class Discriminator(object):
  """docstring for Discriminator"""
  def __init__(self, config, train_inception=False):
    self.config = config
    self.train_inception = train_inception

    self.initializer = tf.random_uniform_initializer(
      minval=-self.config.initializer_scale,
      maxval=self.config.initializer_scale)

    self.images = None
    self.fake_seqs = None
    self.true_seqs = None

    self.image_embeddings = None
    self.fake_seqs_embeddings = None
    self.true_seqs_embeddings = None

    self.init_fn = None
    self.total_loss = None
    self.reward = None

    self.inception_variables = None
    
  def is_training(self):
    return True

  def process_image(self, encoded_image, thread_id=0):
    return drm_image_processing.process_image(encoded_image,
      is_training=self.is_training(),
      height=self.config.image_height,
      width=self.config.image_width,
      thread_id=thread_id,
      image_format=self.config.image_format)

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.
    Inputs:
      self.images
    Outputs:
      self.image_embeddings
    """
    inception_output = drm_image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")
    self.image_embeddings = image_embeddings

  def lookup_seq_embeddings(self, seqs):
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
        name="map",
        shape=[self.config.vocab_size, self.config.embedding_size],
        initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, seqs)

    return seq_embeddings

  def build_seq_embeddings(self):
    self.true_seqs_embeddings = lookup_seq_embeddings(self.true_seqs)
    self.fake_seqs_embeddings = lookup_seq_embeddings(self.fake_seqs)

  def r_eta(a, b):
    """Get the relationship between imageEmbeddding 'a' and 
    sequnenceEmbedding 'b', and map it into (0,1) with sigmoid
    """
    return tf.sigmoid(tf.reduce_sum((a*b)))

  def build_model(self):
    """Based on above varibles, get total_loss
    Inputs:
      self.image_embeddings, A float32 Tensor with 
          shape [batch_size, embeddings] = [32, 512]
      self.fake_seqs_embeddings,
      self.true_seqs_embeddings, A float32 Tensor with 
          shape [batch_size, embeddings] = [32, 512]
    Outputs:
      self.total_loss, A float32 scalar Tensor
    Method:
      \[-\log r(I,S)-\log(1-r(I,g))\]
    """
    batch_loss = -tf.log(r_eta(self.image_embeddings, self.true_seqs_embeddings)) - \
      tf.log(1-r_eta(self.image_embeddings, self.fake_seqs_embeddings))

    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()

    # Reward for Ganerator's Agent
    self.reward = r_eta(self.image_embeddings, self.fake_seqs_embeddings)

    # Add summaries.
    tf.summary.scalar("losses/batch_loss", batch_loss)
    tf.summary.scalar("losses/total_loss", total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram("parameters/" + var.op.name, var)

    self.total_loss = total_loss

  def setup_inception_initializer(self):
    saver = tf.train.Saver(self.inception_variables)
    def restore_fn(sess):
      tf.logging.info("Restoring Inception variables from checkpoint file %s", 
        self.config.inception_checkpoint_file)
      saver.restore(sess, self.config.inception_checkpoint_file)
    self.init_fn = restore_fn

  def build(self):
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()