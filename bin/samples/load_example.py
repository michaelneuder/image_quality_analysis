#!/usr/bin/env python3
import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('ts0.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    print(sess.run('weights1:0'))
