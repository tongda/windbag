import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops

from windbag import config
from windbag.model.decoder import Decoder
from windbag.model.model_base import ChatBotModelBase


class BasicChatBotModel(ChatBotModelBase):
    def __init__(self, batch_size):
        super(ChatBotModelBase, self).__init__()
        self.batch_size = batch_size

        self.encoder_inputs_tensor = tf.placeholder(tf.int32,
                                                    shape=[None, batch_size],
                                                    name='encoder_inputs')
        self.decoder_length_tensor = tf.placeholder(tf.int32, shape=(batch_size,), name='decoder_lens')
        self.target_tensor = tf.placeholder(tf.int32,
                                            shape=[None, batch_size],
                                            name='decoder_inputs')
        self.bucket_length = tf.placeholder(tf.int32, shape=(2,), name='bucket_length')

        self.global_step = tf.contrib.framework.get_global_step()

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
        self.train_op = self.create_loss(self.final_outputs)

        tf.summary.histogram("final-outputs", self.final_outputs)

    def encode(self):
        with tf.variable_scope('encoder') as scope:
            scope = tf.get_variable_scope()
            scope.set_initializer(tf.random_uniform_initializer(-0.1, 0.1))

            W = tf.get_variable(
                name="W",
                shape=[config.ENC_VOCAB, config.HIDDEN_SIZE],
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            source_embedded = tf.nn.embedding_lookup(W, self.encoder_inputs_tensor)

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
            def condition(time, all_outputs, inputs, states):
                return time < self.bucket_length[1] - 1
                # return tf.reduce_all(self.decoder_length_tensor > time)

            def body(time, all_outputs, inputs, state):
                dec_outputs, dec_state, output_logits, next_input = self.decoder.step(inputs, state)
                all_outputs = all_outputs.write(time, output_logits)
                return time + 1, all_outputs, next_input, dec_state

            output_ta = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                     size=0,
                                                     dynamic_size=True,
                                                     element_shape=(self.batch_size, config.DEC_VOCAB))

            print("self.target_tensor[0]: ", self.target_tensor[0])
            res = control_flow_ops.while_loop(
                condition,
                body,
                loop_vars=[0, output_ta, self.decoder.zero_input(self.target_tensor[0]), enc_final_state],
            )
            final_outputs = res[1].stack()
            final_state = res[3]
        return final_outputs, final_state

    def create_loss(self, final_outputs):
        with tf.variable_scope('loss') as scope:
            print("final_outputs: ", final_outputs.get_shape())
            print("self.decoder_inputs_tensor: ", self.target_tensor.get_shape())
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=final_outputs, labels=self.target_tensor[1:])

            loss_mask = tf.sequence_mask(
                tf.to_int32(self.decoder_length_tensor), self.bucket_length[1] - 1)

            losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])
            # self.loss = tf.reduce_mean(losses)
            self.loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(self.decoder_length_tensor - 1))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.LR)
            trainables = tf.trainable_variables()
            self.grads = self.optimizer.compute_gradients(self.loss, trainables)
            tf.summary.scalar("loss", self.loss)
            train_op = self.optimizer.apply_gradients(self.grads, global_step=self.global_step)

        return train_op
