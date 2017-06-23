import tensorflow as tf


class Decoder(object):
    def __init__(self, input_size, hidden_size):
        with tf.variable_scope("decoder") as scope:
            scope = tf.get_variable_scope()
            scope.set_initializer(tf.random_uniform_initializer(-0.1, 0.1))

            self.scope = scope
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            self.embedding = tf.get_variable(
                name="embedding",
                shape=[input_size, hidden_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

    def _lookup(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)

    def step(self, inputs, state):
        with tf.variable_scope(self.scope):
            embedded = self._lookup(inputs=inputs)
            return self.cell(inputs=embedded, state=state)


class AttentionDecoder(Decoder):
    def __init__(self, *args, enc_outputs: tf.Tensor, num_ctx_hidden_units: int):
        super(AttentionDecoder, self).__init__(*args)
        self.enc_outputs = enc_outputs
        self.num_ctx_units = num_ctx_hidden_units

    def step_(self, inputs, state, att_ctx):
        with tf.variable_scope(self.scope):
            embedded = self._lookup(inputs=inputs)
            whole_input = tf.concat([embedded, att_ctx], 1)

            outputs, next_state = self.cell(inputs=whole_input, state=state)

            att_keys = tf.contrib.layers.fully_connected(
                inputs=self.enc_outputs, num_outputs=self.num_ctx_units, activation_fn=None)
            att_query = tf.contrib.layers.fully_connected(
                inputs=outputs, num_outputs=self.num_ctx_units, activation_fn=None)
            # the simplest score function
            scores = tf.reduce_sum(att_keys * tf.expand_dims(att_query, 0), [2])

            scores_normalized = tf.nn.softmax(scores, dim=0)
            context = tf.expand_dims(scores_normalized, 2) * self.enc_outputs
            context = tf.reduce_sum(context, [0], name="context")

        return outputs, next_state, context