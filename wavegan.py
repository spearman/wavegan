import tensorflow as tf

# generator
class G (tf.keras.Model):
  def __init__ (self,
    slice_len=16384,
    nch=1,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    upsample='zeros'):
    super().__init__()
    assert slice_len in [16384, 32768, 65536]
    # in the conversion to TF2, we are only supporting zeros for now
    assert upsample == 'zeros'
    self.slice_len = slice_len
    self.nch = nch
    self.kernel_len = kernel_len
    self.dim = dim
    self.use_batchnorm = use_batchnorm
    self.dim_mul = dim_mul = 16 if slice_len == 16384 else 32
    self.z_project = tf.keras.layers.Dense (4 * 4 * dim * dim_mul)
    dim_mul //= 2
    self.upconv_0 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    dim_mul //= 2
    self.upconv_1 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    dim_mul //= 2
    self.upconv_2 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    dim_mul //= 2
    self.upconv_3 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    if slice_len == 16384:
      self.upconv_4 = tf.keras.layers.Conv1DTranspose (nch, kernel_len, 4,
        padding='same')
    elif slice_len == 32768:
      self.upconv_4 = tf.keras.layers.Conv1DTranspose (dim, kernel_len, 4,
        padding='same')
      self.upconv_5 = tf.keras.layers.Conv1DTranspose (nch, kernel_len, 2,
        padding='same')
    elif slice_len == 65536:
      self.upconv_4 = tf.keras.layers.Conv1DTranspose (dim, kernel_len, 4,
        padding='same')
      self.upconv_5 = tf.keras.layers.Conv1DTranspose (nch, kernel_len, 4,
        padding='same')
  def __call__ (self, z, train):
    return WaveGANGenerator (self, z, train)

def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_width),
        strides=(1, stride),
        padding='same'
        )[:, 0]
  elif upsample == 'nn':
    batch_size = tf.shape(inputs)[0]
    _, w, nch = inputs.get_shape().as_list()

    x = inputs

    x = tf.expand_dims(x, axis=1)
    x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
    x = x[:, 0]

    return tf.layers.conv1d(
        x,
        filters,
        kernel_width,
        1,
        padding='same')
  else:
    raise NotImplementedError


"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def WaveGANGenerator(
    generator,
    z,
    train=False):
  dim_mul = generator.dim_mul
  batch_size = tf.shape(z)[0]

  if generator.use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  #FIXME: debug
  if output == None:
    print ("NONE-2")
  output = generator.z_project (output)   # dense
  output = tf.reshape(output, [batch_size, 16, generator.dim * dim_mul])
  output = batchnorm(output)
  output = tf.nn.relu(output)

  #FIXME: debug
  if output == None:
    print ("NONE-1")

  # Layer 0
  # [16, 1024] -> [64, 512]
  output = generator.upconv_0 (output)
  output = batchnorm(output)
  output = tf.nn.relu(output)

  #FIXME: debug
  if output == None:
    print ("NONE0")

  # Layer 1
  # [64, 512] -> [256, 256]
  output = generator.upconv_1 (output)
  output = batchnorm(output)
  output = tf.nn.relu(output)

  #FIXME: debug
  if output == None:
    print ("NONE1")

  # Layer 2
  # [256, 256] -> [1024, 128]
  output = generator.upconv_2 (output)
  output = batchnorm(output)
  output = tf.nn.relu(output)

  #FIXME: debug
  if output == None:
    print ("NONE2")

  # Layer 3
  # [1024, 128] -> [4096, 64]
  output = generator.upconv_3 (output)
  output = batchnorm(output)
  output = tf.nn.relu(output)

  #FIXME: debug
  if output == None:
    print ("NONE3")

  if generator.slice_len == 16384:
    # Layer 4
    # [4096, 64] -> [16384, nch]
    output = generator.upconv_4 (output)
    output = tf.nn.tanh(output)
  elif generator.slice_len == 32768:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    output = generator.upconv_4 (output)
    output = batchnorm(output)
    output = tf.nn.relu(output)

    # Layer 5
    # [16384, 64] -> [32768, nch]
    output = generator.upconv_5 (output)
    output = tf.nn.tanh(output)
  elif generator.slice_len == 65536:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    output = generator.upconv_4 (output)
    output = batchnorm(output)
    output = tf.nn.relu(output)

    # Layer 5
    # [16384, 64] -> [65536, nch]
    output = generator.upconv_5 (output)
    output = tf.nn.tanh(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and generator.use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
    if slice_len == 16384:
      assert len(update_ops) == 10
    else:
      assert len(update_ops) == 12
    with tf.control_dependencies(update_ops):
      output = tf.identity(output)

  return output


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x

# generator
class D (tf.keras.Model):
  def __init__ (self,
    slice_len,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0):
    super().__init__()
    self.slice_len = slice_len
    self.kernel_len = kernel_len
    self.dim = dim
    self.use_batchnorm = use_batchnorm
    self.phaseshuffle_rad = phaseshuffle_rad
    self.downconv_0 = tf.keras.layers.Conv1D (dim, kernel_len, 4,
      padding='same')
    self.downconv_1 = tf.keras.layers.Conv1D (dim * 2, kernel_len, 4,
      padding='same')
    self.downconv_2 = tf.keras.layers.Conv1D (dim * 4, kernel_len, 4,
      padding='same')
    self.downconv_3 = tf.keras.layers.Conv1D (dim * 8, kernel_len, 4,
      padding='same')
    self.downconv_4 = tf.keras.layers.Conv1D (dim * 16, kernel_len, 4,
      padding='same')
    if slice_len == 32768:
      self.downconv_5 = tf.keras.layers.Conv1D (dim * 32, kernel_len, 2,
        padding='same')
    elif slice_len == 65536:
      self.downconv_5 = tf.keras.layers.Conv1D (dim * 32, kernel_len, 4,
        padding='same')
    self.out = tf.keras.layers.Dense (1)

  def __call__ (self, x):
    return WaveGANDiscriminator (self, x)


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(discriminator, x):
  batch_size = tf.shape(x)[0]
  slice_len = int(x.get_shape()[1])
  assert discriminator.slice_len == slice_len

  if discriminator.use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if discriminator.phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, discriminator.phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  output = discriminator.downconv_0 (output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  output = discriminator.downconv_1 (output)
  output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  output = discriminator.downconv_2 (output)
  output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  output = discriminator.downconv_3 (output)
  output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  output = discriminator.downconv_4 (output)
  output = batchnorm(output)
  output = lrelu(output)

  if slice_len == 32768:
    # Layer 5
    # [32, 1024] -> [16, 2048]
    output = discriminator.downconv_5 (output)
    output = batchnorm(output)
    output = lrelu(output)
  elif slice_len == 65536:
    # Layer 5
    # [64, 1024] -> [16, 2048]
    output = discriminator.downconv_5 (output)
    output = batchnorm(output)
    output = lrelu(output)

  # Flatten
  output = tf.reshape(output, [batch_size, -1])

  # Connect to single logit
  output = discriminator.out (output)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
