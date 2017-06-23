import tensorflow as tf
from windbag.model.basic_model import BasicChatBotModel
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops

from windbag import config
from windbag.model.decoder import AttentionDecoder


class AttentionChatBotModel(BasicChatBotModel):
    def build(self):
        enc_outputs, enc_final_state = self.encode()

        hidden_size = enc_final_state.get_shape().as_list()[1]
        self.decoder = AttentionDecoder(
            config.DEC_VOCAB, hidden_size, enc_outputs=enc_outputs, num_ctx_hidden_units=config.CONTEXT_SIZE)

        self.final_outputs, final_state = self.decode(enc_outputs, enc_final_state)
        self.train_op = self.create_loss(self.final_outputs)

        tf.summary.histogram("final-outputs", self.final_outputs)

    def decode(self, enc_outputs, enc_final_state):
        with tf.variable_scope('decoder') as scope:
            scope = tf.get_variable_scope()
            scope.set_initializer(tf.random_uniform_initializer(-0.1, 0.1))

            def condition(time, all_outputs, inputs, states, cur_ctx):
                return time < self.bucket_length[1] - 1
                # return tf.reduce_all(self.decoder_length_tensor > time)

            def body(time, all_outputs, inputs, state, cur_ctx):
                dec_outputs, dec_state, next_ctx = self.decoder.step_(inputs, state, cur_ctx)

                with tf.variable_scope('output_calculate') as scope:
                    projection_input = tf.concat([dec_outputs, next_ctx], 1)
                    print("projection_input:", projection_input.get_shape())

                    softmax_input = tf.contrib.layers.fully_connected(
                        inputs=projection_input,
                        num_outputs=self.decoder.cell.output_size,
                        activation_fn=tf.nn.tanh,
                        scope="attention_mix")
                    # TODO: move this into decoder
                    output_logits = tf.contrib.layers.fully_connected(
                        inputs=softmax_input, num_outputs=config.DEC_VOCAB, activation_fn=None, scope="logits")
                    print("output_logits:", output_logits.get_shape())
                    all_outputs = all_outputs.write(time, output_logits)

                with tf.variable_scope('next_input_calculate') as scope:
                    next_input = tf.cast(tf.arg_max(output_logits, dimension=1), tf.int32)

                return time + 1, all_outputs, next_input, dec_state, next_ctx

            output_ta = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                     size=0,
                                                     dynamic_size=True,
                                                     element_shape=(self.batch_size, config.DEC_VOCAB))

            zero_ctx = tf.zeros_like(enc_final_state)
            res = control_flow_ops.while_loop(
                condition,
                body,
                loop_vars=[0, output_ta, self.target_tensor[0], enc_final_state, zero_ctx],
            )
            final_outputs = res[1].stack()
            final_state = res[3]
        return final_outputs, final_state
