import argparse
import os

import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner, RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator

from windbag import config
from windbag.data.dataset import CornellMovieDataset
from windbag.model.hook import IteratorInitializerHook
from windbag.model.model import model_fn


def _get_model_dir(parsed_args):
  exp_name = (("attention" if parsed_args.use_attention else "basic") +
              "-step_" + str(parsed_args.num_steps) +
              "-batch_" + str(parsed_args.batch_size) +
              "-lr_" + str(config.LR))
  if parsed_args.tag:
    exp_name += "-" + parsed_args.tag

  return os.path.join(parsed_args.model_dir, exp_name)


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
    train_steps_per_iteration=hparams.steps_per_eval
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
    batch_size=parsed_args.batch_size,
    shuffle=True,
    shuffle_size=256,
    steps_per_eval=parsed_args.steps_per_eval
  )

  learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule="continuous_train_and_eval",
    hparams=params
  )


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
  parser.add_argument("--steps-per-eval", dest="steps_per_eval", type=int, default=None,
                      help="Number of steps between each evaluation,"
                           "`None` by default. if this is `None`, "
                           "evaluation only happens once after train.")
  parser.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                      help="Batch size.")
  return parser


if __name__ == '__main__':
  main()
