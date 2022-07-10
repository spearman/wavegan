import functools
import os
import time

import tensorflow as tf
import numpy as np

import loader
import model

def train(fps, args):
  train_dataset = loader.decode_extract_and_batch(
    fps,
    batch_size=args.train_batch_size,
    slice_len=args.data_slice_len,
    decode_fs=args.data_sample_rate,
    decode_num_channels=args.data_num_channels,
    decode_fast_wav=args.data_fast_wav,
    decode_parallel_calls=4,
    slice_randomize_offset=False if args.data_first_slice else True,
    slice_first_only=args.data_first_slice,
    slice_overlap_ratio=0. if args.data_first_slice else args.data_overlap_ratio,
    slice_pad_end=True if args.data_first_slice else args.data_pad_end,
    repeat=False,
    shuffle=True,
    shuffle_buffer_size=4096,
    prefetch_size=args.train_batch_size * 4,
    prefetch_gpu_num=args.data_prefetch_gpu_num)
    #prefetch_gpu_num=args.data_prefetch_gpu_num)[:, :, 0]

  #FIXME:debug
  print ("train_dataset:", train_dataset)

  slice_len = int (iter (train_dataset).get_next()[:, :, 0].get_shape()[1])
  #slice_len = int(iter (train_dataset).get_next()[:, :, 0].get_shape()[1])
  #FIXME:debug
  print ("slice_len:", slice_len)
  generator = model.Generator (**args.wavegan_g_kwargs)
  discriminator = model.Discriminator (slice_len=slice_len, **args.wavegan_d_kwargs)

  # call with input to build generator model
  noise = tf.random.normal ([args.train_batch_size, args.wavegan_latent_dim])
  generator (noise, train=False)

  # Print G summary
  print('-' * 80)
  print('Generator vars')
  nparams = 0
  for v in generator.trainable_variables:
    v_shape = v.get_shape().as_list()
    v_n = functools.reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

  # call with input to build discriminator model
  discriminator (iter (train_dataset).get_next()[:, :, 0], train=False)

  # Print D summary
  print('-' * 80)
  print('Discriminator vars')
  nparams = 0
  for v in discriminator.trainable_variables:
    v_shape = v.get_shape().as_list()
    v_n = functools.reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

  # Create (recommended) optimizer
  if args.wavegan_loss == 'dcgan':
    G_opt = tf.keras.optimizers.Adam(
      learning_rate=2e-4,
      beta_1=0.5)
    D_opt = tf.keras.optimizers.Adam(
      learning_rate=2e-4,
      beta_1=0.5)
  elif args.wavegan_loss == 'lsgan':
    G_opt = tf.keras.optimizers.RMSprop(
      learning_rate=1e-4)
    D_opt = tf.keras.optimizers.RMSprop(
      learning_rate=1e-4)
  elif args.wavegan_loss == 'wgan':
    G_opt = tf.keras.optimizers.RMSprop(
      learning_rate=5e-5)
    D_opt = tf.keras.optimizers.RMSprop(
      learning_rate=5e-5)
  elif args.wavegan_loss == 'wgan-gp':
    G_opt = tf.keras.optimizers.Adam(
      learning_rate=1e-4,
      beta_1=0.5,
      beta_2=0.9)
    D_opt = tf.keras.optimizers.Adam(
      learning_rate=1e-4,
      beta_1=0.5,
      beta_2=0.9)
  else:
    raise NotImplementedError()

  @tf.function
  def train_discriminator (x):
    noise = tf.random.normal ([args.train_batch_size, args.wavegan_latent_dim])
    with tf.GradientTape() as disc_tape:
      generated = generator (noise, train=True)
      generated_output = discriminator (generated, train=True)
      real_output = discriminator (x, train=True)
      disc_loss = model.discriminator_loss (args, discriminator, x, generated, real_output, generated_output)
    grad_disc = disc_tape.gradient (disc_loss, discriminator.trainable_variables)
    D_opt.apply_gradients (zip (grad_disc, discriminator.trainable_variables))
    return disc_loss

  @tf.function
  def train_generator():
    noise = tf.random.normal ([args.train_batch_size, args.wavegan_latent_dim])
    with tf.GradientTape() as gen_tape:
      generated = generator (noise, train=True)
      generated_output = discriminator (generated, train=True)
      gen_loss = model.generator_loss (args, generated_output)
    grad_gen = gen_tape.gradient (gen_loss, generator.trainable_variables)
    G_opt.apply_gradients (zip (grad_gen, generator.trainable_variables))
    return gen_loss

  def preview (step):
    from scipy.io.wavfile import write as wavwrite

    preview_dir = os.path.join (args.train_dir, "preview")
    if not os.path.isdir (preview_dir):
      os.makedirs (preview_dir)

    noise = tf.random.normal ([args.preview_n, args.wavegan_latent_dim])
    generated = generator (noise, train=False)
    # Flatten batch
    flat_pad = int (args.data_sample_rate / 2)
    generated_padded = tf.pad (generated, [[0, 0], [0, flat_pad], [0, 0]])
    nch = int (generated.get_shape()[-1])
    print ("NCH:", nch)
    generated_flat = tf.reshape (generated_padded, [-1, nch])
    # Encode to int16
    def float_to_int16 (x):
      x_int16 = x * 32767.0
      x_int16 = tf.clip_by_value (x_int16, -32767.0, 32767.0)
      x_int16 = tf.cast (x_int16, tf.int16)
      return x_int16
    generated_int16 = float_to_int16 (generated)
    generated_flat_int16 = float_to_int16 (generated_flat)
    preview_fp = os.path.join (preview_dir,
      "{}.wav".format (str (step).zfill (8)))
    wavwrite (preview_fp, args.data_sample_rate, np.array (generated_flat_int16))

  epoch = 0
  while True:
    start = time.time()
    disc_loss = 0.0
    gen_loss = 0.0
    sample = 0

    # Train discriminator
    for x in train_dataset:
      sample += 1
      print ("sample:", sample)
      print ("D_opt.iterations:", D_opt.iterations.numpy())
      disc_loss += train_discriminator (x[:, :, 0])
      if D_opt.iterations.numpy() % args.wavegan_disc_nupdates == 0:
        gen_loss += train_generator()
    print ("Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}".format(
      epoch, time.time() - start, gen_loss / args.train_batch_size,
      disc_loss / (args.train_batch_size * args.wavegan_disc_nupdates)))

    epoch += 1

    print ("Generating preview")
    preview (epoch)


if __name__ == '__main__':
  import argparse
  import glob
  import sys

  print ("physical devices: ", tf.config.list_physical_devices())

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
  parser.add_argument('train_dir', type=str,
    help='Training directory')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
    help='Data directory containing *only* audio files to load')
  data_args.add_argument('--data_sample_rate', type=int,
    help='Number of audio samples per second')
  data_args.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],
    help='Number of audio samples per slice (maximum generation length)')
  data_args.add_argument('--data_num_channels', type=int,
    help='Number of audio channels to generate (for >2, must match that of data)')
  data_args.add_argument('--data_overlap_ratio', type=float,
    help='Overlap ratio [0, 1) between slices')
  data_args.add_argument('--data_first_slice', action='store_true', dest='data_first_slice',
    help='If set, only use the first slice each audio example')
  data_args.add_argument('--data_pad_end', action='store_true', dest='data_pad_end',
    help='If set, use zero-padded partial slices from the end of each audio file')
  data_args.add_argument('--data_normalize', action='store_true', dest='data_normalize',
    help='If set, normalize the training examples')
  data_args.add_argument('--data_fast_wav', action='store_true', dest='data_fast_wav',
    help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
  data_args.add_argument('--data_prefetch_gpu_num', type=int,
    help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')

  wavegan_args = parser.add_argument_group('WaveGAN')
  wavegan_args.add_argument('--wavegan_latent_dim', type=int,
    help='Number of dimensions of the latent space')
  wavegan_args.add_argument('--wavegan_kernel_len', type=int,
    help='Length of 1D filter kernels')
  wavegan_args.add_argument('--wavegan_dim', type=int,
    help='Dimensionality multiplier for model of G and D')
  wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
    help='Enable batchnorm')
  wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
    help='Number of discriminator updates per generator update')
  wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
    help='Which GAN loss to use')
  wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn'],
    help='Generator upsample strategy')
  wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
    help='If set, use post-processing filter')
  wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
    help='Length of post-processing filter for DCGAN')
  wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
    help='Radius of phase shuffle operation')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
    help='Batch size')
  train_args.add_argument('--train_save_secs', type=int,
    help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
    help='How often to report summaries')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_n', type=int,
    help='Number of samples to preview')

  incept_args = parser.add_argument_group('Incept')
  incept_args.add_argument('--incept_metagraph_fp', type=str,
    help='Inference model for inception score')
  incept_args.add_argument('--incept_ckpt_fp', type=str,
    help='Checkpoint for inference model')
  incept_args.add_argument('--incept_n', type=int,
    help='Number of generated examples to test')
  incept_args.add_argument('--incept_k', type=int,
    help='Number of groups to test')

  parser.set_defaults(
    data_dir=None,
    data_sample_rate=16000,
    data_slice_len=16384,
    data_num_channels=1,
    data_overlap_ratio=0.,
    data_first_slice=False,
    data_pad_end=False,
    data_normalize=False,
    data_fast_wav=False,
    data_prefetch_gpu_num=0,
    wavegan_latent_dim=100,
    wavegan_kernel_len=25,
    wavegan_dim=64,
    wavegan_batchnorm=False,
    wavegan_disc_nupdates=5,
    wavegan_loss='wgan-gp',
    wavegan_genr_upsample='zeros',
    wavegan_genr_pp=False,
    wavegan_genr_pp_len=512,
    wavegan_disc_phaseshuffle=2,
    train_batch_size=64,
    train_save_secs=300,
    train_summary_secs=120,
    preview_n=32,
    incept_metagraph_fp='./eval/inception/infer.meta',
    incept_ckpt_fp='./eval/inception/best_acc-103005',
    incept_n=5000,
    incept_k=10)

  args = parser.parse_args()

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Save args
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  # Make model kwarg dicts
  setattr(args, 'wavegan_g_kwargs', {
    'batch_size': args.train_batch_size,
    'slice_len': args.data_slice_len,
    'nch': args.data_num_channels,
    'kernel_len': args.wavegan_kernel_len,
    'dim': args.wavegan_dim,
    'use_batchnorm': args.wavegan_batchnorm,
    'upsample': args.wavegan_genr_upsample,
    'pp_filt': args.wavegan_genr_pp,
    'pp_len': args.wavegan_genr_pp_len
  })
  setattr(args, 'wavegan_d_kwargs', {
    'kernel_len': args.wavegan_kernel_len,
    'dim': args.wavegan_dim,
    'use_batchnorm': args.wavegan_batchnorm,
    'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
  })

  if args.mode == 'train':
    fps = glob.glob(os.path.join(args.data_dir, '*'))
    if len(fps) == 0:
      raise Exception('Did not find any audio files in specified directory')
    print('Found {} audio files in specified directory'.format(len(fps)))
    #infer(args)
    train(fps, args)
  elif args.mode == 'preview':
    preview(args)
  elif args.mode == 'incept':
    incept(args)
  elif args.mode == 'infer':
    infer(args)
  else:
    raise NotImplementedError()
