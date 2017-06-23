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
        embedded = self._lookup(inputs=inputs)
        return self.cell(inputs=embedded, state=state)
