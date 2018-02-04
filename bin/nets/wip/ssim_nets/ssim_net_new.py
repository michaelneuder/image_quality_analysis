#!/usr/bin/env python3
import os
import matplotlib as mpl
mpl.use('Agg'); os.environ['TF_CPP_MIN_LOG_LEVEL']='2';
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.gridspec as gridspec
import iqa_tools

def main():
    print('welcome to ssim net ...')

    # network parameters
    filter_dim, filter_dim2 = 11, 1; batch_size = 4;
    image_dim, result_dim = 96, 86
    input_layer, first_layer, second_layer, third_layer, fourth_layer, output_layer = 4, 100, 50, 25, 10, 1
    learning_rate = .001; epochs = 5000

    data_path = '/home/neuder/Dropbox/github/image_quality_analysis/data/sample_data/'
    train_features, train_target, test_features, test_target = iqa_tools.load_data(local=True, path=data_path)
    print('images loaded ...')

    # initializing filters, this is what we are trying to learn --- fan in
    scaling_factor = 0.1
    initializer = tf.contrib.layers.xavier_initializer()
    weights = {
        'weights1': tf.get_variable('weights1', [filter_dim,filter_dim,input_layer,first_layer], initializer=initializer),
        'weights2': tf.get_variable('weights2', [filter_dim2,filter_dim2,first_layer,second_layer], initializer=initializer),
        'weights3': tf.get_variable('weights3', [filter_dim2,filter_dim2,second_layer,third_layer], initializer=initializer),
        'weights4': tf.get_variable('weights4', [filter_dim2,filter_dim2,third_layer,fourth_layer], initializer=initializer),
        'weights_out': tf.get_variable('weights_out', [filter_dim2,filter_dim2,fourth_layer+third_layer+second_layer+first_layer,output_layer], initializer=initializer)
    }
    biases = {
        'bias1': tf.get_variable('bias1', [first_layer], initializer=initializer),
        'bias2': tf.get_variable('bias2', [second_layer], initializer=initializer),
        'bias3': tf.get_variable('bias3', [third_layer], initializer=initializer),
        'bias4': tf.get_variable('bias4', [fourth_layer], initializer=initializer),
        'bias_out': tf.get_variable('bias_out', [output_layer], initializer=initializer)
    }
    print('filters created ...')

    # tensorflow setup
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, input_layer])
    y = tf.placeholder(tf.float32, [None, result_dim, result_dim, output_layer])

    # model
    prediction = iqa_tools.conv_net(x, weights, biases)

    # get variance to normalize error terms during training
    variance = np.var(train_target)

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # error arrays
    training_error, testing_error = [], []

    # saving state of weights
    saver = tf.train.Saver()
    print('tensorflow graph created ...')

    # tensorflow session & training
    with tf.Session() as sess:
        sess.run(init)
        global_start_time = time.time()
        print('starting training...')
        for epoch_count in range(epochs):
            start_time = time.time()
            epoch = iqa_tools.get_epoch(train_features, train_target, batch_size)
            for batch in epoch:
                x_data_train, y_data_train = np.asarray(epoch[batch][0]), np.asarray(epoch[batch][1])
                sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
            train_loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
            training_error.append(100*train_loss/variance)
            test_loss = sess.run(cost, feed_dict={x : test_features, y : test_target})
            testing_error.append(100*test_loss/variance)
            end_time = time.time()
            print('current epoch: {} -- '.format(epoch_count)
                  +'current train error: {:.4f} -- '.format(100*train_loss/variance)
                  +'epoch time: {:.4}s '.format(end_time-start_time))

            # plot the current error
            f, axarr = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
            axarr.plot(np.arange(epoch_count+1), training_error, label='train')
            axarr.plot(np.arange(epoch_count+1), testing_error, label='test')
            axarr.legend()
            axarr.set_ylim(0,100)
            plt.savefig('ssim_net2.png')

            if epoch_count % 100 == 0:
                saver.save(sess, 'saved_nets/ssimNET{}'.format(epoch_count))

    print('training finished.')


if __name__ == '__main__':
    main()
