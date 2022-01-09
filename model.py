import tensorflow as tf

class Generator (tf.keras.Model):
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
    # layers
    self.z_project = tf.keras.layers.Dense (4 * 4 * dim * dim_mul)
    self.reshape_z = tf.keras.layers.Reshape ((None, 16, dim * dim_mul))
    if use_batchnorm:
      self.batchnorm_z = tf.keras.layers.BatchNormalization()
    self.relu_z = tf.keras.layers.ReLU()

    dim_mul //= 2
    self.upconv_0 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    if use_batchnorm:
      self.batchnorm_0 = tf.keras.layers.BatchNormalization()
    self.relu_0 = tf.keras.layers.ReLU()

    dim_mul //= 2
    self.upconv_1 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    if use_batchnorm:
      self.batchnorm_1 = tf.keras.layers.BatchNormalization()
    self.relu_1 = tf.keras.layers.ReLU()

    dim_mul //= 2
    self.upconv_2 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    if use_batchnorm:
      self.batchnorm_2 = tf.keras.layers.BatchNormalization()
    self.relu_2 = tf.keras.layers.ReLU()

    dim_mul //= 2
    self.upconv_3 = tf.keras.layers.Conv1DTranspose (dim * dim_mul, kernel_len,
      4, padding='same')
    if use_batchnorm:
      self.batchnorm_3 = tf.keras.layers.BatchNormalization()
    self.relu_3 = tf.keras.layers.ReLU()

    if slice_len == 16384:
      self.upconv_4 = tf.keras.layers.Conv1DTranspose (nch, kernel_len, 4,
        padding='same', activation='tanh')
    elif slice_len == 32768:
      self.upconv_4 = tf.keras.layers.Conv1DTranspose (dim, kernel_len, 4,
        padding='same')
      if use_batchnorm:
        self.batchnorm_4 = tf.keras.layers.BatchNormalization()
      self.relu_4 = tf.keras.layers.ReLU()
      self.upconv_5 = tf.keras.layers.Conv1DTranspose (nch, kernel_len, 2,
        padding='same', activation='tanh')
    elif slice_len == 65536:
      self.upconv_4 = tf.keras.layers.Conv1DTranspose (dim, kernel_len, 4,
        padding='same')
      if use_batchnorm:
        self.batchnorm_4 = tf.keras.layers.BatchNormalization()
      self.relu_4 = tf.keras.layers.ReLU()
      self.upconv_5 = tf.keras.layers.Conv1DTranspose (nch, kernel_len, 4,
        padding='same', activation='tanh')

  def call (self, z, train):
    dim_mul = self.dim_mul
    batch_size = tf.shape(z)[0]

    # FC and reshape for convolution
    # [100] -> [16, 1024]
    output = z
    output = self.z_project (output)   # dense
    output = self.reshape_z (output, input_shape=[batch_size, 16, self.dim * dim_mul])
    if self.use_batchnorm:
      output = self.batchnorm_z(output, train)
    output = tf.nn.relu(output)

    # Layer 0
    # [16, 1024] -> [64, 512]
    output = self.upconv_0 (output)
    if self.use_batchnorm:
      output = self.batchnorm_0(output, train)
    output = tf.nn.relu(output)

    # Layer 1
    # [64, 512] -> [256, 256]
    output = self.upconv_1 (output)
    if self.use_batchnorm:
      output = self.batchnorm_1(output, train)
    output = tf.nn.relu(output)

    # Layer 2
    # [256, 256] -> [1024, 128]
    output = self.upconv_2 (output)
    if self.use_batchnorm:
      output = self.batchnorm_2(output, train)
    output = tf.nn.relu(output)

    # Layer 3
    # [1024, 128] -> [4096, 64]
    output = self.upconv_3 (output)
    if self.use_batchnorm:
      output = self.batchnorm_3(output, train)
    output = tf.nn.relu(output)

    if self.slice_len == 16384:
      # Layer 4
      # [4096, 64] -> [16384, nch]
      output = self.upconv_4 (output)
      if self.use_batchnorm:
        output = self.batchnorm_4(output, train)
      output = tf.nn.tanh(output)
    elif self.slice_len == 32768:
      # Layer 4
      # [4096, 128] -> [16384, 64]
      output = self.upconv_4 (output)
      if self.use_batchnorm:
        output = self.batchnorm_4(output, train)
      output = tf.nn.relu(output)

      # Layer 5
      # [16384, 64] -> [32768, nch]
      output = self.upconv_5 (output)
      output = tf.nn.tanh(output)
    elif self.slice_len == 65536:
      # Layer 4
      # [4096, 128] -> [16384, 64]
      output = self.upconv_4 (output)
      if self.use_batchnorm:
        output = self.batchnorm_4(output, train)
      output = tf.nn.relu(output)

      # Layer 5
      # [16384, 64] -> [65536, nch]
      output = self.upconv_5 (output)
      output = tf.nn.tanh(output)

    return output

class Discriminator (tf.keras.Model):
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

    # layers
    self.downconv_0 = tf.keras.layers.Conv1D (dim, kernel_len, 4,
      padding='same')
    self.lrelu_0 = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.downconv_1 = tf.keras.layers.Conv1D (dim * 2, kernel_len, 4,
      padding='same')
    if use_batchnorm:
      self.batchnorm_1 = tf.keras.layers.BatchNormalization()
    self.lrelu_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.downconv_2 = tf.keras.layers.Conv1D (dim * 4, kernel_len, 4,
      padding='same')
    if use_batchnorm:
      self.batchnorm_2 = tf.keras.layers.BatchNormalization()
    self.lrelu_2 = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.downconv_3 = tf.keras.layers.Conv1D (dim * 8, kernel_len, 4,
      padding='same')
    if use_batchnorm:
      self.batchnorm_3 = tf.keras.layers.BatchNormalization()
    self.lrelu_3 = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.downconv_4 = tf.keras.layers.Conv1D (dim * 16, kernel_len, 4,
      padding='same')
    if use_batchnorm:
      self.batchnorm_4 = tf.keras.layers.BatchNormalization()
    self.lrelu_4 = tf.keras.layers.LeakyReLU(alpha=0.2)

    if slice_len == 32768:
      self.downconv_5 = tf.keras.layers.Conv1D (dim * 32, kernel_len, 2,
        padding='same')
    elif slice_len == 65536:
      self.downconv_5 = tf.keras.layers.Conv1D (dim * 32, kernel_len, 4,
        padding='same')
    if use_batchnorm:
      self.batchnorm_5 = tf.keras.layers.BatchNormalization()
    self.lrelu_5 = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.flatten = tf.keras.layers.Flatten()

    self.out = tf.keras.layers.Dense (1)

  def call (self, x, train):
    batch_size = tf.shape(x)[0]
    slice_len = int(x.get_shape()[1])
    assert self.slice_len == slice_len

    if self.phaseshuffle_rad > 0:
      # TODO: phaseshuffle
      assert False
      phaseshuffle = lambda x: apply_phaseshuffle(x, self.phaseshuffle_rad)
    else:
      phaseshuffle = lambda x: x

    # Layer 0
    # [16384, 1] -> [4096, 64]
    output = x
    output = self.downconv_0 (output)
    output = self.lrelu_0(output)
    output = phaseshuffle(output)

    # Layer 1
    # [4096, 64] -> [1024, 128]
    output = self.downconv_1 (output)
    if self.use_batchnorm:
      output = self.batchnorm_1(output, train)
    output = self.lrelu_1(output)
    output = phaseshuffle(output)

    # Layer 2
    # [1024, 128] -> [256, 256]
    output = self.downconv_2 (output)
    if self.use_batchnorm:
      output = self.batchnorm_2(output, train)
    output = self.lrelu_2(output)
    output = phaseshuffle(output)

    # Layer 3
    # [256, 256] -> [64, 512]
    output = self.downconv_3 (output)
    if self.use_batchnorm:
      output = self.batchnorm_3(output, train)
    output = self.lrelu_3(output)
    output = phaseshuffle(output)

    # Layer 4
    # [64, 512] -> [16, 1024]
    output = self.downconv_4 (output)
    if self.use_batchnorm:
      output = self.batchnorm_4(output, train)
    output = self.lrelu_4(output)

    if slice_len == 32768:
      # Layer 5
      # [32, 1024] -> [16, 2048]
      output = self.downconv_5 (output)
      if self.use_batchnorm:
        output = self.batchnorm_5(output, train)
      output = self.lrelu_5(output)
    elif slice_len == 65536:
      # Layer 5
      # [64, 1024] -> [16, 2048]
      output = self.downconv_5 (output)
      if self.use_batchnorm:
        output = self.batchnorm_5(output, train)
      output = self.lrelu_5(output)

    # Flatten
    output = self.flatten (output)
    #output = tf.reshape(output, [batch_size, -1])

    # Connect to single logit
    output = self.out (output)[:, 0]

    return output
