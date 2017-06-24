import tensorflow as tf


class Decoder(object):
    def __init__(self, vocab_size, hidden_size):
        with tf.variable_scope("decoder") as scope:
            scope = tf.get_variable_scope()
            scope.set_initializer(tf.random_uniform_initializer(-0.1, 0.1))

            self.scope = scope
            self.vocab_size = vocab_size
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            self.embedding = tf.get_variable(
                name="embedding",
                shape=[vocab_size, hidden_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

    def _lookup(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)

    def step(self, inputs, state):
        with tf.variable_scope(self.scope):
            embedded = self._lookup(inputs=inputs)
            outputs, nxt_state = self.cell(inputs=embedded, state=state)
            logits = tf.contrib.layers.fully_connected(
                inputs=outputs, num_outputs=self.vocab_size, activation_fn=None)
            next_input = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            return outputs, nxt_state, logits, next_input

    def zero_input(self, inputs):
        return inputs


class AttentionDecoder(Decoder):
    def __init__(self, *args, enc_outputs: tf.Tensor, num_ctx_hidden_units: int):
        super(AttentionDecoder, self).__init__(*args)
        self.enc_outputs = enc_outputs
        self.num_ctx_units = num_ctx_hidden_units

    def step(self, inputs, state):
        with tf.variable_scope(self.scope):

            outputs, next_state = self.cell(inputs=inputs, state=state)
            scores_normalized = self._compute_scores(outputs)

            context = tf.expand_dims(scores_normalized, 2) * self.enc_outputs
            context = tf.reduce_sum(context, [0], name="context")

            logits = self._compute_logits(context, outputs)
            next_input = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            next_input = self._compute_next_input(next_input, context)

        return outputs, next_state, logits, next_input

    def zero_input(self, inputs):
        zero_ctx = tf.zeros(dtype=tf.float32, shape=self.enc_outputs.get_shape()[1:])
        return self._compute_next_input(inputs, zero_ctx)

    def _compute_next_input(self, inputs, context):
        embedded = self._lookup(inputs=inputs)
        return tf.concat([embedded, context], 1)

    def _compute_logits(self, context, outputs):
        projection_input = tf.concat([outputs, context], 1)
        projection_input = tf.contrib.layers.fully_connected(
            inputs=projection_input,
            num_outputs=self.cell.output_size,
            activation_fn=tf.nn.tanh,
            scope="attention_mix")
        logits = tf.contrib.layers.fully_connected(
            inputs=projection_input, num_outputs=self.vocab_size, activation_fn=None, scope="logits")
        return logits

    def _compute_scores(self, outputs):
        att_keys = tf.contrib.layers.fully_connected(
            inputs=self.enc_outputs, num_outputs=self.num_ctx_units, activation_fn=None)
        att_query = tf.contrib.layers.fully_connected(
            inputs=outputs, num_outputs=self.num_ctx_units, activation_fn=None)
        # the simplest score function
        scores = tf.reduce_sum(att_keys * tf.expand_dims(att_query, 0), [2])
        scores_normalized = tf.nn.softmax(scores, dim=0)
        return scores_normalized
