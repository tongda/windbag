import tensorflow as tf
import numpy as np

from windbag import config
from windbag.model.basic_model import BasicChatBotModel


class BasicModelTestCase(tf.test.TestCase):
    def test_decoder_lookup_should_return_correct_dimension(self):
        model = BasicChatBotModel(batch_size=2)
        embedded = model._decoder_input_lookup(tf.constant([1, 2]))
        self.assertEqual((2, config.HIDDEN_SIZE), embedded.get_shape())


if __name__ == '__main__':
    tf.test.main()
