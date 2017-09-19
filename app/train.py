import argparse
import os
import random

import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner, RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys
from tensorflow.python.training.session_run_hook import SessionRunHook

from windbag import config
from windbag.data import cornell_movie
from windbag.data.dataset import CornellMovieDataset
from windbag.model.attention_model import AttentionChatBotModel
from windbag.model.basic_model import BasicChatBotModel, create_loss


def need_print_log(step):
  if step < 100:
    return step % 10 == 0
  else:
    return step % 200 == 0


def _get_random_bucket(train_buckets_scale):
  """ Get a random bucket from which to choose a training sample """
  rand = random.random()
  return min([i for i in range(len(train_buckets_scale))
              if train_buckets_scale[i] > rand])


def model_fn(features, labels, mode, params, config):
  model = AttentionChatBotModel(features)
  model.build()
  predictions = tf.argmax(model.final_outputs, axis=-1)
  loss_op = None
  train_op = None

  if mode != ModeKeys.PREDICT:
    loss_op, train_op = create_loss(model.final_outputs, labels)

  return EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss_op,
    train_op=train_op
    # eval_metric_ops={"Accuracy": tf.metrics.accuracy(labels=labels['answer'], predictions=predictions, name='accuracy')}
  )


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


def train_by_estimator(output_dir, train_steps):
  run_config = RunConfig()
  run_config = run_config.replace(model_dir=output_dir)

  params = HParams(
    learning_rate=config.LR,
    train_steps=train_steps,
    min_eval_frequency=100
  )

  learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule="train_and_evaluate",
    hparams=params
  )


def train(use_attention, num_steps=1000, ckpt_dir="./ckp-dir/", write_summary=True, tag=None):
  if not os.path.exists(config.PROCESSED_PATH):
    cornell_movie.prepare_raw_data()
    cornell_movie.process_data()

  if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

  dataset = CornellMovieDataset(config.PROCESSED_PATH, config.BATCH_SIZE, None, config.BATCH_SIZE, True)

  if not use_attention:
    model = BasicChatBotModel(features=dataset.train_features)
  else:
    model = AttentionChatBotModel(features=dataset.train_features)
  model.build()

  loss_op, train_op = create_loss(model.final_outputs, dataset.train_features)

  cfg = tf.ConfigProto()
  cfg.gpu_options.allow_growth = True
  with tf.Session(config=cfg) as sess:
    saver = tf.train.Saver()
    log_root = "./logs/"
    exp_name = (("attention" if use_attention else "basic") +
                "-step_" + str(num_steps) +
                "-batch_" + str(config.BATCH_SIZE) +
                "-lr_" + str(config.LR))
    if tag:
      exp_name += "-" + tag
    summary_writer = tf.summary.FileWriter(log_root + exp_name, graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(num_steps + 1):
      output_logits, res_loss, _ = sess.run([model.final_outputs, loss_op, train_op])
      if need_print_log(step):
        print("Iteration {} - loss:{}".format(step, res_loss))
        if write_summary:
          summaries = sess.run(model.summaries)
          summary_writer.add_summary(summaries, step)
        saver.save(sess, ckpt_dir + exp_name + "/checkpoints", global_step=step)


def create_parser():
  parser = argparse.ArgumentParser(description="Windbag trainer.")
  parser.add_argument("--use-attention", dest="use_attention", action="store_true",
                      help="Flag of whether to use attention.")
  parser.add_argument("--num-steps", dest="num_steps", type=int, default=10,
                      help="Number of steps.")
  parser.add_argument("--write-summary", dest="write_summary", action="store_true",
                      help="Flag of whether to write summaries.")
  parser.add_argument("--tag", dest="tag", type=str, default=None,
                      help="Tag of experiment.")
  return parser


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)

  parser = create_parser()
  args = parser.parse_args()
  # train(args.use_attention, num_steps=args.num_steps, write_summary=args.write_summary, tag=args.tag)
  exp_name = (("attention" if args.use_attention else "basic") +
              "-step_" + str(args.num_steps) +
              "-batch_" + str(config.BATCH_SIZE) +
              "-lr_" + str(config.LR))
  if args.tag:
    exp_name += "-" + args.tag
  train_by_estimator(os.path.join("./ckp-dir/", exp_name), args.num_steps)
