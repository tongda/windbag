import argparse
import os

import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner, RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys
from tensorflow.python.training.session_run_hook import SessionRunHook

from windbag import config
from windbag.data.dataset import CornellMovieDataset
from windbag.model.attention_model import AttentionChatBotModel
from windbag.model.basic_model import create_loss


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


def model_fn(features, labels, mode, params, config):
  questions_and_answers = bucketing(
    questions=features,
    answers=labels,
    boundaries=params.buckets,
    batch_size=params.batch_size,
    shuffle=params.shuffle,
    shuffle_size=params.shuffle_size
  )

  qlen_it = q_it = alen_it = a_it = None

  if mode == ModeKeys.TRAIN:
    qlen_it, q_it, alen_it, a_it = questions_and_answers \
      .repeat() \
      .make_one_shot_iterator() \
      .get_next()

  if mode == ModeKeys.EVAL:
    iterator = questions_and_answers \
      .make_initializable_iterator()
    qlen_it, q_it, alen_it, a_it = iterator.get_next()
    tf.add_to_collection("initializer", iterator.initializer)

  model = AttentionChatBotModel(q_it)
  model.build()
  predictions = tf.argmax(model.final_outputs, axis=-1)
  loss_op = None
  train_op = None

  if mode != ModeKeys.PREDICT:
    loss_op = create_loss(model.final_outputs, a_it, alen_it)
    train_op = _get_train_op(loss_op, params.learning_rate)

  return EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss_op,
    train_op=train_op
    # eval_metric_ops={"Accuracy": tf.metrics.accuracy(labels=labels['answer'], predictions=predictions, name='accuracy')}
  )


def _get_train_op(loss_op, learning_rate):
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  trainables = tf.trainable_variables()
  grads = optimizer.compute_gradients(loss_op, trainables)
  tf.summary.scalar("loss", loss_op)
  global_step = tf.contrib.framework.get_global_step()
  train_op = optimizer.apply_gradients(grads, global_step=global_step)
  return train_op


class IteratorInitializerHook(SessionRunHook):
  def __init__(self):
    super(IteratorInitializerHook, self).__init__()
    self.iterator_initializer = None

  def after_create_session(self, session, coord):
    for initializer in tf.get_collection("initializer"):
      session.run(initializer)


def experiment_fn(run_config, hparams):
  eval_initializer_hook = IteratorInitializerHook()

  estimator = Estimator(
    model_fn=model_fn,
    params=hparams,
    config=run_config)

  train_dataset = CornellMovieDataset(
    os.path.join(config.PROCESSED_PATH, "train_ids.enc"),
    os.path.join(config.PROCESSED_PATH, "train_ids.dec"),
    os.path.join(config.PROCESSED_PATH, "vocab.enc"),
    os.path.join(config.PROCESSED_PATH, "vocab.dec"),
  )

  test_dataset = CornellMovieDataset(
    os.path.join(config.PROCESSED_PATH, "test_ids.enc"),
    os.path.join(config.PROCESSED_PATH, "test_ids.dec"),
    os.path.join(config.PROCESSED_PATH, "vocab.enc"),
    os.path.join(config.PROCESSED_PATH, "vocab.dec"),
  )

  experiment = Experiment(
    estimator=estimator,
    train_input_fn=train_dataset.input_fn,
    eval_input_fn=test_dataset.input_fn,
    train_steps=hparams.train_steps,
    eval_hooks=[eval_initializer_hook],
    eval_steps=None,
  )
  return experiment


def main():
  tf.logging.set_verbosity(tf.logging.DEBUG)

  parsed_args = get_parser().parse_args()

  model_dir = _get_model_dir(parsed_args)

  run_config = RunConfig()
  run_config = run_config.replace(model_dir=model_dir)

  params = HParams(
    learning_rate=parsed_args.lr,
    train_steps=parsed_args.num_steps,
    buckets=config.BUCKETS,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    shuffle_size=256,
  )

  learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule="train_and_evaluate",
    hparams=params
  )


def _get_model_dir(parsed_args):
  exp_name = (("attention" if parsed_args.use_attention else "basic") +
              "-step_" + str(parsed_args.num_steps) +
              "-batch_" + str(config.BATCH_SIZE) +
              "-lr_" + str(config.LR))
  if parsed_args.tag:
    exp_name += "-" + parsed_args.tag

  return os.path.join(parsed_args.model_dir, exp_name)


def get_parser():
  parser = argparse.ArgumentParser(description="Windbag trainer.")
  parser.add_argument("--use-attention", dest="use_attention", action="store_true",
                      help="Flag of whether to use attention.")
  parser.add_argument("--num-steps", dest="num_steps", type=int, default=10,
                      help="Number of steps.")
  parser.add_argument("--write-summary", dest="write_summary", action="store_true",
                      help="Flag of whether to write summaries.")
  parser.add_argument("--tag", dest="tag", type=str, default=None,
                      help="Tag of experiment.")
  parser.add_argument("--learning-rate", dest="lr", type=float, default=0.001,
                      help="Define initial learning rate.")
  parser.add_argument("--model-dir", dest="model_dir", type=str, default="./ckp-dir/",
                      help="define where to save model. "
                           "This is the root dir, every run of experiment will have "
                           "its own sub dir with name generated internally.")
  return parser


if __name__ == '__main__':
  main()
