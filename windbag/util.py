import functools

import tensorflow as tf
from tensorflow.contrib.data import Dataset

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


def bucketing(questions,
              answers,
              boundaries,
              batch_size,
              shuffle,
              shuffle_size):
  '''
  Bucketing questions and answers for training.

  :param questions:
  :param answers:
  :param boundaries:
  :param batch_size:
  :param shuffle:
  :param shuffle_size:
  :return:
   Tuple of (question length, question, answer length, answer)
  '''

  def _which_bucket(question_len, question, answer_len, answer):
    q_max_boundaries, a_max_boundaries = list(zip(*boundaries))
    which_bucket = tf.reduce_min(
      tf.where(tf.logical_and(
        question_len <= q_max_boundaries,
        answer_len <= a_max_boundaries
      ))
    )
    return tf.to_int64(which_bucket)

  def _reduce_batch(key, batch):
    return batch.padded_batch(batch_size, ((), (None,), (), (None,)))

  q_max, a_max = max(boundaries)
  questions_and_answers = Dataset.zip((
    questions.map(lambda q: tf.size(q)),
    questions,
    answers.map(lambda a: tf.size(a)),
    answers,
  )).filter(lambda q_size, q, a_size, a: tf.logical_and(q_size <= q_max, a_size <= a_max))
  questions_and_answers = questions_and_answers.group_by_window(
    _which_bucket, _reduce_batch, batch_size)
  if shuffle:
    questions_and_answers = questions_and_answers.shuffle(shuffle_size)
  return questions_and_answers