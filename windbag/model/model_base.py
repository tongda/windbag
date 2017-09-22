import abc


class ChatBotModelBase(metaclass=abc.ABCMeta):
  def __init__(self, features):
    self.features = features

  @abc.abstractmethod
  def encode(self):
    raise NotImplemented

  @abc.abstractmethod
  def decode(self, enc_outputs, enc_final_state):
    raise NotImplemented

  @abc.abstractmethod
  def build(self):
    raise NotImplemented
