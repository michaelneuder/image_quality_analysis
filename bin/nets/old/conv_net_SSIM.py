#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time
from PIL import Image as im

def convolve_inner_layers(x, W, b):
    y = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return tf.nn.tanh(y)

def convolve_ouput_layer(x, W, b):
    y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return y

def conv_net(x, W, b):
    conv1 = convolve_inner_layers(x, W['weights1'], b['bias1'])
    conv2 = convolve_inner_layers(conv1, W['weights2'], b['bias2'])
    conv3 = convolve_inner_layers(conv2, W['weights3'], b['bias3'])
    output = convolve_ouput_layer(conv3, W['weights_out'], b['bias_out'])
    return output

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

    filter_dim = 11

    weights = {
        'weights1': tf.Variable((1/(filter_dim*filter_dim*2))*tf.random_normal([filter_dim,filter_dim,2,30])),
        'weights2': tf.Variable((1/(30*filter_dim*filter_dim))*tf.random_normal([filter_dim,filter_dim,30,20])),
        'weights3': tf.Variable((1/(20*filter_dim*filter_dim))*tf.random_normal([filter_dim,filter_dim,20,10])),
        'weights_out': tf.Variable((1/(10*filter_dim*filter_dim))*tf.random_normal([filter_dim,filter_dim,10,1]))
    }
    biases = {
        'bias1': tf.Variable((1/(filter_dim*filter_dim*2))*tf.random_normal([30])),
        'bias2': tf.Variable((1/(30*filter_dim*filter_dim))*tf.random_normal([20])),
        'bias3': tf.Variable((1/(20*filter_dim*filter_dim))*tf.random_normal([10])),
        'bias_out': tf.Variable((1/(10*filter_dim*filter_dim))*tf.random_normal([1]))
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
    combined_data_train = np.reshape(np.dstack((original_images_train, reconstructed_images_train)), [train_size,96,96,2])
    combined_data_test =  np.reshape(np.dstack((original_images_test, reconstructed_images_test)), [test_size,96,96,2])

    #### temporary edit --- don't forget to remove
    for i in range(96,192):
        print(original_images_train[0][i], reconstructed_images_train[0][i], combined_data_train[0][1][i-96])
    exit()

    # paramaters
    learning_rate = .0001
    epochs = 100

    # model
    prediction = conv_net(x, weights, biases)

    # saving state
    saver = tf.train.Saver()

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # session
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
            epoch = get_epoch(combined_data_train, comparison_images_train, 50)
            for i in epoch:
                x_data_train, y_data_train = np.asarray(epoch[i][0]), np.asarray(epoch[i][1])
                sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
                loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
                print("  -  training global_step {}. current error: {}. ".format(global_step, loss))
                global_step+=1
            print('epoch {} completed in {} seconds. current error = {}'.format(epoch_count, time.time()-epoch_time, loss))
            print('-------------------------------------------------------')
            epoch_count+=1
        print('optimization finished!')

        prediction = np.asarray(sess.run(prediction, feed_dict={x : [combined_data_train[0]]}))
        target = np.asarray([comparison_images_test[0]])
        print(prediction.shape, target.shape)
        with open('post_training.csv', mode = 'w') as write_file:
            write_file.write('target, prediction\n')
            for i in range(96):
                for j in range(96):
                    write_file.write(str(float(target[0][i][j][0])) + ', ' + str(float(prediction[0][i][j][0])) + '\n')
        write_file.close()

if __name__ == '__main__':
    main()
