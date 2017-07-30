#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time
from PIL import Image as im

def convolve_inner_layers(x, W, b):
    x = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)

def convolve_ouput_layer(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

def conv_net(x, W, b):
    conv1 = convolve_inner_layers(x, W['weights1'], b['bias1'])
    conv2 = convolve_inner_layers(conv1, W['weights2'], b['bias2'])
    conv3 = convolve_inner_layers(conv2, W['weights3'], b['bias3'])
    conv4 = convolve_inner_layers(conv3, W['weights4'], b['bias4'])
    output = convolve_ouput_layer(conv4, W['weights_out'], b['bias_out'])
    return output

def get_batch(x, y, n):
    batch_indices = np.arange(x.shape[0])
    np.random.shuffle(batch_indices)
    x_batch = []
    y_batch = []
    for i in batch_indices[:n]:
        x_batch.append(x[i])
        y_batch.append(y[i])
    return [x_batch, y_batch]

def main():
    print("--------------------------------------------------")
    print("initializing variables ...")
    weights = {
        'weights1': tf.get_variable('weights1', shape=[11,11,2,30], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'weights2': tf.get_variable('weights2', shape=[11,11,30,15], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'weights3': tf.get_variable('weights3', shape=[11,11,15,7], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'weights4': tf.get_variable('weights4', shape=[11,11,7,3], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'weights_out': tf.get_variable('weights_out', shape=[11,11,3,1], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN'))
    }
    biases = {
        'bias1': tf.get_variable('bias1', shape=[30], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'bias2': tf.get_variable('bias2', shape=[15], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'bias3': tf.get_variable('bias3', shape=[7], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'bias4': tf.get_variable('bias4', shape=[3], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
        'bias_out': tf.get_variable('bias_out', shape=[1], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN'))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 96, 96, 2])
    y = tf.placeholder(tf.float32, [None, 96, 96, 1])

    # data
    print("loading data ...")
    original_images_train = np.loadtxt('../../../data/sample_data/orig_50.txt')
    reconstructed_images_train = np.loadtxt('../../../data/sample_data/recon_50.txt')
    comparison_images_train = np.loadtxt('../../../data/sample_data/comp_50.txt')
    original_images_test = np.loadtxt('../../../data/sample_data/orig_15.txt')
    reconstructed_images_test = np.loadtxt('../../../data/sample_data/recon_15.txt')
    comparison_images_test = np.loadtxt('../../../data/sample_data/comp_15.txt')

    # get size of training and testing set
    train_size = original_images_train.shape[0]
    test_size = original_images_test.shape[0]

    # reshaping the result data to --- (num pics), 96, 96, 1
    comparison_images_train = np.reshape(comparison_images_train, [train_size, 96, 96, 1])
    comparison_images_test = np.reshape(comparison_images_test, [test_size, 96, 96, 1])

    # zipping data
    input_combined_train = []
    for i in range(train_size):
        for j in range(96*96):
            input_combined_train.append([original_images_train[i][j], reconstructed_images_train[i][j]])
    input_combined_train = np.asarray(input_combined_train, dtype=np.float32)
    input_combined_train = np.reshape(input_combined_train, [train_size, 96,96, 2])
    input_combined_test = []
    for i in range(test_size):
        for j in range(96*96):
            input_combined_test.append([original_images_test[i][j], reconstructed_images_test[i][j]])
    input_combined_test = np.asarray(input_combined_test, dtype=np.float32)
    input_combined_test = np.reshape(input_combined_test, [test_size, 96,96, 2])
    test = input_combined_train[0]
    image1 = np.asarray(test[0], dtype='uint8')
    image2 = np.asarray(test[1], dtype='uint8')
    image3 = np.asarray(np.reshape(comparison_images_train[0],[96,96]), dtype='uint8')
    image_view = im.fromarray(image1, 'L')
    image_view.show()
    image_view = im.fromarray(image2, 'L')
    image_view.show()
    image_view = im.fromarray(image3, 'L')
    image_view.show()

    # paramaters
    learning_rate = .0001
    training_iterations = 10000

    # model
    prediction = conv_net(x, weights, biases)

    # saving state
    saver = tf.train.Saver()

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # evaluation
    error = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))

    init = tf.global_variables_initializer()
    with open('../../../data/results/loss_lr{}_trn{}_tst{}.csv'.format(learning_rate, train_size, test_size), mode='w') as write_file:
        with tf.Session() as sess:
            sess.run(init)
            step = 0
            start_time = time.time()
            print("starting training ... ")
            write_file.write('step, error\n')
            while step < training_iterations:
                x_data_train, y_data_train = [input_combined_train[0]], [comparison_images_train[0]]
                sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
                loss = sess.run(error, feed_dict={x : x_data_train, y : y_data_train})
                print("training step {}. current error: {}. ".format(step, loss))
                write_file.write(str(step)+', '+str(loss)+'\n')
                step += 1
            print("optimization finished!")
            print("--------------------------------------------------")
            print("testing accuracy")
            x_data_test, y_data_test = [input_combined_test[0]], [comparison_images_test[0]]
            final_error = sess.run(error, feed_dict={x: x_data_test, y: y_data_test})
            print("the average pixel difference on the test set is {}.".format(final_error))
            print("--------------------------------------------------")
            print('training took {} seconds'.format(time.time()- start_time))
    write_file.close()

if __name__ == '__main__':
    main()
