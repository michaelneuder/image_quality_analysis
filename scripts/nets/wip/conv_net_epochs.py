#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time
import datetime

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
    output = convolve_ouput_layer(conv3, W['weights_out'], b['bias_out'])
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

def get_epoch(x, y, n):
    input_size = x.shape[0]
    number_batches = int(input_size / n)
    extra_examples = input_size % n
    batches = {}
    batch_indices = np.arange(input_size)
    np.random.shuffle(batch_indices)
    for i in range(number_batches):
        temp_indices = batch_indices[n*i:n*(i+1)]
        temp_x = []
        temp_y = []
        for j in temp_indices:
            temp_x.append(x[j])
            temp_y.append(y[j])
        batches[i] = [np.asarray(temp_x), np.asarray(temp_y)]
    if extra_examples != 0:
        extra_indices = batch_indices[input_size-extra_examples:input_size]
        temp_x = []
        temp_y = []
        for k in extra_indices:
            temp_x.append(x[k])
            temp_y.append(y[k])
        batches[i+1] = [np.asarray(temp_x), np.asarray(temp_y)]
    return batches

def main():
    # a bit of ascii fun
    print('                            _       _   _             ')
    print('   ___ ___  _ ____   _____ | |_   _| |_(_) ___  _ __  ')
    print('  / __/ _ \| \'_ \ \ / / _ \| | | | | __| |/ _ \| \'_ \ ')
    print(' | (_| (_) | | | \ V / (_) | | |_| | |_| | (_) | | | |')
    print('  \___\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_|')
    print('=======================================================')
    print("initializing variables ...")
    # weights = {
    #     'weights1': tf.get_variable('weights1', shape=[11,11,2,30], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'weights2': tf.get_variable('weights2', shape=[11,11,30,15], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'weights3': tf.get_variable('weights3', shape=[11,11,15,7], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'weights4': tf.get_variable('weights4', shape=[11,11,7,3], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'weights_out': tf.get_variable('weights_out', shape=[11,11,3,1], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN'))
    # }
    # biases = {
    #     'bias1': tf.get_variable('bias1', shape=[30], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'bias2': tf.get_variable('bias2', shape=[15], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'bias3': tf.get_variable('bias3', shape=[7], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'bias4': tf.get_variable('bias4', shape=[3], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN')),
    #     'bias_out': tf.get_variable('bias_out', shape=[1], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1,mode='FAN_IN'))
    # }

    weights = {
        'weights1': tf.Variable((1/(11*11*10*2))*tf.random_normal([11,11,2,30])),
        'weights2': tf.Variable((1/(11*11*10*30))*tf.random_normal([11,11,30,20])),
        'weights3': tf.Variable((1/(11*11*10*20))*tf.random_normal([11,11,20,10])),
        'weights_out': tf.Variable((1/(11*11*10*10))*tf.random_normal([11,11,10,1]))
    }
    biases = {
        'bias1': tf.Variable((1/(11*11*10*30))*tf.random_normal([30])),
        'bias2': tf.Variable((1/(11*11*10*20))*tf.random_normal([20])),
        'bias3': tf.Variable((1/(11*11*10*10))*tf.random_normal([10])),
        'bias_out': tf.Variable((1/(11*11*10*1))*tf.random_normal([1]))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 96, 96, 2])
    y = tf.placeholder(tf.float32, [None, 96, 96, 1])

    # data
    print("loading data ...")
    original_images_train = np.loadtxt('../../../data/sample_data/orig_500.txt')
    reconstructed_images_train = np.loadtxt('../../../data/sample_data/recon_500.txt')
    comparison_images_train = np.loadtxt('../../../data/sample_data/comp_500.txt')
    original_images_test = np.loadtxt('../../../data/sample_data/orig_140.txt')
    reconstructed_images_test = np.loadtxt('../../../data/sample_data/recon_140.txt')
    comparison_images_test = np.loadtxt('../../../data/sample_data/comp_140.txt')

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

    # paramaters
    learning_rate = .00001
    epochs = 100

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
    with tf.Session() as sess:
        sess.run(init)
        global_step = 0
        epoch_count = 0
        start_time = time.time()
        print("starting training ... ")
        while epoch_count < epochs:
            epoch_time = time.time()
            print('-------------------------------------------------------')
            print('beginning epoch {} ...'.format(epoch_count))
            epoch = get_epoch(input_combined_train, comparison_images_train, 50)
            for i in epoch:
                x_data_train, y_data_train = epoch[i][0], epoch[i][1]
                sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
                loss = sess.run(error, feed_dict={x : x_data_train, y : y_data_train})
                print("  -  training global_step {}. current error: {}. ".format(global_step, loss))
                global_step+=1
                current_error = loss
            print('epoch {} completed in {} seconds. current error = {}'.format(epoch_count, time.time()-epoch_time ,loss))
            print('-------------------------------------------------------')
            epoch_count+=1
        print('optimization finished!')

if __name__ == '__main__':
    main()
