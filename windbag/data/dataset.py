import abc
import os

import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset, Dataset
from windbag import config


def _build_dict(vocab_path):
  with (open(vocab_path, encoding='ascii', errors='replace')) as f:
    words = [line.strip() for line in f]
    return dict(zip(range(len(words)), words))


def _read_id_file(path) -> Dataset:
  def _parse_line(line):
    splits = tf.string_split(tf.reshape(line, (-1,))).values
    return tf.string_to_number(splits, out_type=tf.int32)

  return TextLineDataset(path) \
    .filter(lambda line: tf.size(line) > 0) \
    .map(_parse_line)


class WindbagDataset(metaclass=abc.ABCMeta):
  @property
  def train_features(self):
    raise NotImplementedError

  @property
  def test_features(self):
    raise NotImplementedError

  @property
  def question_vocab_mapping(self):
    raise NotImplementedError

  @property
  def answer_vocab_mapping(self):
    raise NotImplementedError


class CornellMovieDataset(WindbagDataset):
  FEATURE_NAMES = ("question_len", "question", "answer_len", "answer")

  def __init__(self,
               data_dir,
               train_batch_size,
               train_epochs,
               test_batch_size,
               shuffle=False):
    self.data_dir = data_dir
    self._question_vocab_mapping = None
    self._answer_vocab_mapping = None

    train_dataset = self._build_datasets("train",
                                         batch_size=train_batch_size,
                                         shuffle=shuffle)
    train_features = train_dataset.repeat(train_epochs).make_one_shot_iterator().get_next()
    self._train_features = dict(zip(CornellMovieDataset.FEATURE_NAMES,
                                    train_features))

    # test data should be reinitialized every time it is used
    self._test_dataset = self._build_datasets("test",
                                              batch_size=test_batch_size,
                                              shuffle=False)

  def _build_datasets(self, dataset_prefix, batch_size, shuffle, shuffle_size=1024):
    question_path = os.path.join(self.data_dir, "%s_ids.enc" % dataset_prefix)
    answer_path = os.path.join(self.data_dir, "%s_ids.dec" % dataset_prefix)
    question_dataset = _read_id_file(question_path)
    answer_dataset = _read_id_file(answer_path)

    bucket_boundaries = config.BUCKETS

    def _which_bucket(question_len, question, answer_len, answer):
      q_max_boundaries, a_max_boundaries = list(zip(*bucket_boundaries))
      which_bucket = tf.reduce_min(
        tf.where(tf.logical_and(
          question_len <= q_max_boundaries,
          answer_len <= a_max_boundaries
        ))
      )
      return tf.to_int64(which_bucket)

    def _reduce_batch(key, batch):
      return batch.padded_batch(batch_size, ((), (None,), (), (None,)))

    questions_and_answers = Dataset.zip((
      question_dataset.map(lambda q: tf.size(q)),
      question_dataset,
      answer_dataset.map(lambda a: tf.size(a)),
      answer_dataset,
    ))
    questions_and_answers = questions_and_answers.group_by_window(
      _which_bucket, _reduce_batch, batch_size)
    if shuffle:
      questions_and_answers = questions_and_answers.shuffle(shuffle_size)
    return questions_and_answers

  @property
  def question_vocab_mapping(self):
    if self._question_vocab_mapping is None:
      enc_vocab = os.path.join(self.data_dir, "vocab.enc")
      self._question_vocab_mapping = _build_dict(enc_vocab)
    return self._question_vocab_mapping

  @property
  def answer_vocab_mapping(self):
    if self._answer_vocab_mapping is None:
      dec_vocab = os.path.join(self.data_dir, "vocab.dec")
      self._answer_vocab_mapping = _build_dict(dec_vocab)
    return self._answer_vocab_mapping

  @property
  def train_features(self):
    return self._train_features

  @property
  def test_features(self):
    test_iterator = self._test_dataset.make_initializable_iterator()
    features = test_iterator.get_next()

    test_features = dict(zip(CornellMovieDataset.FEATURE_NAMES, features))

    return test_iterator.initializer(), test_features
