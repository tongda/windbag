import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops

from windbag import config
from windbag.model.decoder import Decoder
from windbag.model.model_base import ChatBotModelBase

ANSWER_START = 2
ANSWER_END = 3
ANSWER_MAX = 60


def create_loss(final_outputs, labels):
  '''
  Final outputs of the decoder may have different length with
  target answer. So we should pad the outputs if the outputs
  are shorter than target answer, and pad the target answers
  if outputs are longer than answers.
  :param final_outputs: the output of decoder
  :param labels: the target answers
  :return: tuple of loss_op and train_op
  '''
  with tf.variable_scope('loss') as scope:
    target_lens_tensor = labels['answer_len']
    target_tensor = tf.transpose(labels['answer'], (1, 0))
    print("target_tensor[0]: ", target_tensor[0])
    print("final_outputs: ", final_outputs.get_shape())
    print("decoder_inputs_tensor: ", target_tensor.get_shape())

    answer_max_len = tf.reduce_max(target_lens_tensor)
    output_len = tf.shape(final_outputs)[0]

    def loss_with_padded_outputs():
      indexes = [[0, 1]]
      values = tf.expand_dims(answer_max_len - output_len - 1, axis=0)
      # because rank of final outputs tensor is 3, so the shape is (3, 2)
      shape = [3, 2]
      paddings = tf.scatter_nd(indexes, values, shape)
      padded_outputs = tf.pad(final_outputs, paddings)
      return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=padded_outputs, labels=target_tensor[1:])

    def loss_with_padded_answers():
      indexes = [[0, 1]]
      values = tf.expand_dims(output_len - answer_max_len + 1, axis=0)
      # because rank of answers tensor is 2, so the shape is (2, 2)
      shape = [2, 2]
      paddings = tf.scatter_nd(indexes, values, shape)
      padded_answer = tf.pad(target_tensor, paddings)
      return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=final_outputs, labels=padded_answer[1:])

    losses = tf.cond(output_len < answer_max_len, loss_with_padded_outputs, loss_with_padded_answers)

    losses_length = tf.shape(losses)[0]
    loss_mask = tf.sequence_mask(
      tf.to_int32(target_lens_tensor), losses_length)

    losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])
    # self.loss = tf.reduce_mean(losses)
    loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(target_lens_tensor - 1))

  return loss


class BasicChatBotModel(ChatBotModelBase):
  def __init__(self, features):
    super(BasicChatBotModel, self).__init__(features)

    self.source_tensor = tf.transpose(features['question'], (1, 0))

  @property
  def summaries(self, use_all=False):
    if use_all:
      return tf.summary.merge_all()
    else:
      return tf.summary.merge(['loss'])

  def build(self):
    enc_outputs, enc_final_state = self.encode()

    hidden_size = enc_final_state.get_shape().as_list()[1]
    self.decoder = Decoder(config.DEC_VOCAB, hidden_size)

    self.final_outputs, final_state = self.decode(enc_outputs, enc_final_state)

    tf.summary.histogram("final-outputs", self.final_outputs)

  def encode(self):
    with tf.variable_scope('encoder') as scope:
      scope = tf.get_variable_scope()
      scope.set_initializer(tf.random_uniform_initializer(-0.1, 0.1))

      W = tf.get_variable(
        name="W",
        shape=[config.ENC_VOCAB, config.HIDDEN_SIZE],
        initializer=tf.random_uniform_initializer(-0.1, 0.1))

      source_embedded = tf.nn.embedding_lookup(W, self.source_tensor)

      cell_fw = tf.nn.rnn_cell.GRUCell(num_units=config.HIDDEN_SIZE)
      cell_bw = tf.nn.rnn_cell.GRUCell(num_units=config.HIDDEN_SIZE)

      # cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=config.HIDDEN_SIZE) for _ in range(config.NUM_LAYERS)])
      enc_outputs, enc_final_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, source_embedded, time_major=True, dtype=tf.float32)

      enc_outputs = tf.concat(enc_outputs, 2)
      enc_final_state = tf.concat(enc_final_state, 1)
      print("enc_final_state: ", enc_final_state.get_shape())

      # enc_outputs, enc_final_state = tf.nn.dynamic_rnn(cell=cell,
      #                                                  inputs=source_embedded,
      #                                                  time_major=True,
      #                                                  dtype=tf.float32)
    return enc_outputs, enc_final_state

  def decode(self, enc_outputs, enc_final_state):
    with tf.variable_scope(self.decoder.scope):
      def condition(time, all_outputs: tf.TensorArray, inputs, states):
        # bucket_length = tf.shape(self.target_tensor)[0]
        all_ends = tf.reduce_all(all_outputs.read(time) == ANSWER_END)
        return tf.logical_and(tf.logical_not(all_ends), tf.less(time, ANSWER_MAX))
        # return tf.reduce_all(self.decoder_length_tensor > time)

      def body(time, all_outputs, inputs, state):
        dec_outputs, dec_state, output_logits, next_input = self.decoder.step(inputs, state)
        all_outputs = all_outputs.write(time, output_logits)
        return time + 1, all_outputs, next_input, dec_state

      output_ta = tensor_array_ops.TensorArray(dtype=tf.float32,
                                               size=0,
                                               dynamic_size=True,
                                               element_shape=(None, config.DEC_VOCAB))

      # with time-major data input, the batch size is the second dimension
      batch_size = tf.shape(enc_outputs)[1]
      # zero_input = tf.ones(tf.concat([tf.expand_dims(batch_size, axis=0), [1]], axis=0), dtype=tf.int32) * ANSWER_START
      zero_input = tf.ones(tf.expand_dims(batch_size, axis=0), dtype=tf.int32) * ANSWER_START
      res = control_flow_ops.while_loop(
        condition,
        body,
        loop_vars=[0, output_ta, self.decoder.zero_input(zero_input), enc_final_state],
      )
      final_outputs = res[1].stack()
      final_state = res[3]
    return final_outputs, final_state
