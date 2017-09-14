import abc


class ChatBotModelBase(metaclass=abc.ABCMeta):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    @abc.abstractmethod
    def encode(self):
        raise NotImplemented

    @abc.abstractmethod
    def decode(self, enc_outputs, enc_final_state):
        raise NotImplemented

    @abc.abstractmethod
    def build(self):
        raise NotImplemented

    @abc.abstractmethod
    def create_loss(self, final_outputs):
        raise NotImplemented
