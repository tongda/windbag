import argparse
import random
import os

import numpy as np
import tensorflow as tf

from windbag import config
from windbag.data import cornell_movie
from windbag.model.attention_model import AttentionChatBotModel
from windbag.model.basic_model import BasicChatBotModel
from windbag.util import get_buckets


def need_print_log(step):
    if step < 100:
        return step % 10 == 0
    else:
        return step % 200 == 0


def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


def train(use_attention, num_steps=1000, ckpt_dir="./ckp-dir/", write_summary=True, tag=None):
    if not os.path.exists(config.PROCESSED_PATH):
        cornell_movie.prepare_raw_data()
        cornell_movie.process_data()

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    test_buckets, data_buckets, train_buckets_scale = get_buckets()

    if not use_attention:
        model = BasicChatBotModel(batch_size=config.BATCH_SIZE)
    else:
        model = AttentionChatBotModel(batch_size=config.BATCH_SIZE)
    model.build()

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        saver = tf.train.Saver()
        log_root = "./logs/"
        exp_name = (("attention" if use_attention else "basic") +
                    "-step_" + str(num_steps) +
                    "-batch_" + str(config.BATCH_SIZE) +
                    "-lr_" + str(config.LR))
        if tag:
            exp_name += "-" + tag
        summary_writer = tf.summary.FileWriter(log_root + exp_name, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        for step in range(num_steps + 1):
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = cornell_movie.get_batch(
                data_buckets[bucket_id], bucket_id, batch_size=config.BATCH_SIZE)
            decoder_lens = np.sum(np.transpose(np.array(decoder_masks), (1, 0)), axis=1)
            feed_dict = {model.encoder_inputs_tensor: encoder_inputs, model.target_tensor: decoder_inputs,
                         model.decoder_length_tensor: decoder_lens,
                         model.bucket_length: config.BUCKETS[bucket_id]}
            output_logits, res_loss, _ = sess.run([model.final_outputs, model.loss, model.train_op],
                                                  feed_dict=feed_dict)

            if need_print_log(step):
                print("Iteration {} - loss:{}".format(step, res_loss))
                if write_summary:
                    summaries = sess.run(model.summaries, feed_dict=feed_dict)
                    summary_writer.add_summary(summaries, step)
                saver.save(sess, ckpt_dir + exp_name + "/checkpoints", global_step=step)


def create_parser():
    parser = argparse.ArgumentParser(description="Windbag trainer.")
    parser.add_argument("--use-attention", dest="use_attention", action="store_true",
                        help="Flag of whether to use attention.")
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=10,
                        help="Number of steps.")
    parser.add_argument("--write-summary", dest="write_summary", action="store_true",
                        help="Flag of whether to write summaries.")
    parser.add_argument("--tag", dest="tag", type=str, default=None,
                        help="Tag of experiment.")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    train(args.use_attention, num_steps=args.num_steps, write_summary=args.write_summary, tag=args.tag)
