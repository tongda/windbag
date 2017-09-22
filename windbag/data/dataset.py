import abc
import os

import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset, Dataset


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


class CorpusBase(metaclass=abc.ABCMeta):
  @property
  def question_vocab_mapping(self):
    raise NotImplementedError

  @property
  def answer_vocab_mapping(self):
    raise NotImplementedError

  @abc.abstractmethod
  def input_fn(self):
    raise NotImplementedError


class CornellMovieDataset(CorpusBase):
  def input_fn(self):
    return self._build_datasets()

  FEATURE_NAMES = ("question_len", "question", "answer_len", "answer")

  def __init__(self, questions_path, answers_path, question_vocab_path, answer_vocab_path):
    self._questions_path = questions_path
    self._answers_path = answers_path
    self._question_vocab_path = question_vocab_path
    self._answer_vocab_path = answer_vocab_path

    self._question_vocab_mapping = None
    self._answer_vocab_mapping = None

  def _build_datasets(self):
    question_dataset = _read_id_file(self._questions_path)
    answer_dataset = _read_id_file(self._answers_path)

    return question_dataset, answer_dataset

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
