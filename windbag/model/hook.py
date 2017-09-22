import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook


class IteratorInitializerHook(SessionRunHook):
  def __init__(self):
    super(IteratorInitializerHook, self).__init__()
    self.iterator_initializer = None

  def after_create_session(self, session, coord):
    for initializer in tf.get_collection("initializer"):
      session.run(initializer)