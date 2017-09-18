import os
import sys

from windbag import config
from windbag.data import cornell_movie
import numpy as np
import tensorflow as tf
from windbag.model.basic_model import BasicChatBotModel
from windbag.model.attention_model import AttentionChatBotModel


def _check_restore_parameters(sess, saver, ckpt_path):
  """ Restore the previously trained parameters if there are any. """
  ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path))
  if ckpt and ckpt.model_checkpoint_path:
    print("Loading parameters for the Chatbot")
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print("Initializing fresh parameters for the Chatbot")


def _get_user_input():
  """ Get user's input, which will be transformed into encoder input later """
  print("> ", end="")
  sys.stdout.flush()
  return sys.stdin.readline().encode('ascii')


def _construct_response(output_logits, inv_dec_vocab):
  """ Construct a response to the user's encoder input.
  @output_logits: the outputs from sequence to sequence wrapper.
  output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

  This is a greedy decoder - outputs are just argmaxes of output_logits.
  """
  print("logits: ", output_logits[0])
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits[0]]
  print(outputs)
  # If there is an EOS symbol in outputs, cut them at that point.
  if config.EOS_ID in outputs:
    outputs = outputs[:outputs.index(config.EOS_ID)]
  # Print out sentence corresponding to outputs.
  return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


def chat(use_attention, ckpt_path="./ckp-dir/checkpoints"):
  """ in test mode, we don't to create the backward path
  """
  _, enc_vocab = cornell_movie.load_vocab(
    os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
  inv_dec_vocab, _ = cornell_movie.load_vocab(
    os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

  question = tf.placeholder(dtype=tf.int32, shape=(1, None))

  if not use_attention:
    model = BasicChatBotModel(batch_size=1)
  else:
    # TODO: here is a hot fix to pass features into model, should be refactored
    model = AttentionChatBotModel(
      features={'question': question, 'answer_len': tf.constant([0]), 'answer': question},
      targets=None,
      batch_size=1
    )
  model.build()

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _check_restore_parameters(sess, saver, ckpt_path)
    output_file = open(os.path.join(
      config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
    # Decode from standard input.
    max_length = config.BUCKETS[-1][0]
    print(
      'Welcome to TensorBro. Say something. Enter to exit. Max length is', max_length)
    while True:
      line = _get_user_input()
      if len(line) > 0 and line[-1] == b'\n':
        line = line[:-1]
      if line == b'':
        break
      output_file.write('HUMAN ++++ ' + line.decode('ascii') + '\n')
      # Get token-ids for the input sentence.
      token_ids = cornell_movie.sentence2id(enc_vocab, line)
      if len(token_ids) > max_length:
        print('Max length I can handle is:', max_length)
        continue

      # Get output logits for the sentence.
      output_logits = sess.run([model.final_outputs],
                               feed_dict={question: [token_ids]})
      response = _construct_response(output_logits, inv_dec_vocab)
      print(response)
      output_file.write('BOT ++++ ' + response + '\n')
    output_file.write('=============================================\n')
    output_file.close()


if __name__ == '__main__':
  chat(True, ckpt_path="./ckp-dir/attention-step_10-batch_2-lr_0.001/checkpoints")
