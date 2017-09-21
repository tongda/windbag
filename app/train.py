import argparse
import os

import tensorflow as tf
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


def model_fn(features, labels, mode, params, config):
  model = AttentionChatBotModel(features)
  model.build()
  predictions = tf.argmax(model.final_outputs, axis=-1)
  loss_op = None
  train_op = None

  if mode != ModeKeys.PREDICT:
    loss_op = create_loss(model.final_outputs, labels)
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
    session.run(self.iterator_initializer)


def experiment_fn(run_config, hparams):
  estimator = Estimator(
    model_fn=model_fn,
    params=hparams,
    config=run_config)

  # dataset = CornellMovieDataset(config.PROCESSED_PATH, config.BATCH_SIZE, None, config.BATCH_SIZE, True)

  def train_input_fn():
    dataset = CornellMovieDataset(config.PROCESSED_PATH, config.BATCH_SIZE, None, config.BATCH_SIZE, True)
    return dataset.train_features, dataset.train_features

  eval_initializer_hook = IteratorInitializerHook()

  def test_input_fn():
    dataset = CornellMovieDataset(config.PROCESSED_PATH, config.BATCH_SIZE, None, config.BATCH_SIZE, True)
    initializer, features = dataset.test_features
    eval_initializer_hook.iterator_initializer = initializer
    return features, features

  experiment = Experiment(
    estimator=estimator,
    train_input_fn=train_input_fn,
    eval_input_fn=test_input_fn,
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
    min_eval_frequency=100
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
