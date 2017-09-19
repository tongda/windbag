import tensorflow as tf

from windbag import config
from windbag.model.basic_model import BasicChatBotModel
from windbag.model.decoder import AttentionDecoder


class AttentionChatBotModel(BasicChatBotModel):
  def build(self):
    enc_outputs, enc_final_state = self.encode()

    hidden_size = enc_final_state.get_shape().as_list()[1]
    self.decoder = AttentionDecoder(
      config.DEC_VOCAB, hidden_size, enc_outputs=enc_outputs, num_ctx_hidden_units=config.CONTEXT_SIZE)

    self.final_outputs, final_state = self.decode(enc_outputs, enc_final_state)

    tf.summary.histogram("final-outputs", self.final_outputs)
