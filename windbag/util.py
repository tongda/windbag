import functools

import tensorflow as tf
from windbag.data import cornell_movie

from windbag import config


def get_buckets():
  """ Load the dataset into buckets based on their lengths.
  train_buckets_scale is the inverval that'll help us
  choose a random bucket later on.
  """
  test_buckets = cornell_movie.load_data('test_ids.enc', 'test_ids.dec')
  data_buckets = cornell_movie.load_data('train_ids.enc', 'train_ids.dec')
  train_bucket_sizes = [len(data_buckets[b])
                        for b in range(len(config.BUCKETS))]
  print("Number of samples in each bucket:\n", train_bucket_sizes)
  train_total_size = sum(train_bucket_sizes)
  # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]
  print("Bucket scale:\n", train_buckets_scale)
  return test_buckets, data_buckets, train_buckets_scale


def ready_for_reuse(name):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      temp_func = tf.make_template(name, func)
      return temp_func(*args, **kwargs)

    return wrapper

  return decorator
