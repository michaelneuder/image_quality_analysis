#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time
import pandas as pd

def convolve_inner_layers(x, W, b):
    y = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return tf.nn.tanh(y)

def convolve_ouput_layer(x, W, b):
    y = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return y

def conv_net(x, W, b):
    conv1 = convolve_inner_layers(x, W['weights1'], b['bias1'])
    conv2 = convolve_inner_layers(conv1, W['weights2'], b['bias2'])
    conv3 = convolve_inner_layers(conv2, W['weights3'], b['bias3'])
    output_feed = tf.concat([conv1, conv2, conv3],3)
    output = convolve_ouput_layer(output_feed, W['weights_out'], b['bias_out'])
    return output

def get_variance(training_target):
    all_pixels = training_target.flatten()
    return all_pixels.var()

def get_epoch(x, y, n):
    input_size = x.shape[0]
    number_batches = input_size // n
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

def normalize_input(train_data, test_data):
    mean, std_dev = np.mean(train_data, axis=0), np.std(train_data, axis=0)
    return (train_data - mean) / std_dev, (test_data - mean) / std_dev

def main():
    # parameters
    filter_dim = 11
    filter_dim2 = 1
    batch_size = 4
    image_dim = 96
    input_layer = 2
    first_layer = 50
    second_layer = 25
    third_layer = 10
    output_layer = 1
    initializer_scale = 10.0
    learning_rate = .00001
    epochs = 400

    # seeding for debug purposes --- dont forget to remove
    SEED = 12345
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    print('loading image files ... ')
    # train/test images
    orig_500 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/orig_500.txt', header=None, delim_whitespace = True)
    recon_500 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/recon_500.txt', header=None, delim_whitespace = True)
    SSIM_500 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/SSIM_500.txt', header=None, delim_whitespace = True)
    orig_140 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/orig_140.txt', header=None, delim_whitespace = True)
    recon_140 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/recon_140.txt', header=None, delim_whitespace = True)
    SSIM_140 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/SSIM_140.txt', header=None, delim_whitespace = True)

    # normaliztion
    original_images_train = orig_500.values
    reconstructed_images_train = recon_500.values
    original_images_test = orig_140.values
    reconstructed_images_test = recon_140.values

    training_input = np.dstack((original_images_train, reconstructed_images_train))
    testing_input = np.dstack((original_images_test, reconstructed_images_test))

    training_input_normalized, testing_input_normalized = normalize_input(training_input, testing_input)
    comparison_images_train = SSIM_500.values
    comparison_images_test = SSIM_140.values

    # get size of training and testing set
    train_size = original_images_train.shape[0]
    test_size = original_images_test.shape[0]

    # reshaping the result data to --- (num pics), 96, 96, 1
    target_data_train = np.reshape(comparison_images_train, [train_size, image_dim, image_dim, 1])
    target_data_test = np.reshape(comparison_images_test, [test_size, image_dim, image_dim, 1])

    # reshaping
    train_data = np.reshape(training_input_normalized, [train_size,image_dim,image_dim,2])
    test_data =  np.reshape(testing_input_normalized, [test_size,image_dim,image_dim,2])

    # initializing variables --- fan in
    scaling_factor = 1.0
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=scaling_factor, mode='FAN_IN')
    weights = {
        'weights1': tf.get_variable('weights1', [filter_dim,filter_dim,input_layer,first_layer], initializer=initializer),
        'weights2': tf.get_variable('weights2', [filter_dim2,filter_dim2,first_layer,second_layer], initializer=initializer),
        'weights3': tf.get_variable('weights3', [filter_dim2,filter_dim2,second_layer,third_layer], initializer=initializer),
        'weights_out': tf.get_variable('weights4', [filter_dim2,filter_dim2,third_layer+second_layer+first_layer,output_layer], initializer=initializer)
    }
    biases = {
        'bias1': tf.get_variable('bias1', [first_layer], initializer=initializer),
        'bias2': tf.get_variable('bias2', [second_layer], initializer=initializer),
        'bias3': tf.get_variable('bias3', [third_layer], initializer=initializer),
        'bias_out': tf.get_variable('bias4', [output_layer], initializer=initializer)
    }


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, 2])
    y = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])

    # model
    prediction = conv_net(x, weights, biases)

    # get variance to normalize error terms during training
    variance = get_variance(target_data_train)

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        epoch_count = 0
        global_step = 0
        start_time = time.time()
        print("starting training ... ")
        while epoch_count < epochs:
            print('---------------------------------------------------------')
            print('beginning epoch {} ...'.format(epoch_count))
            epoch = get_epoch(train_data, target_data_train, batch_size)
            for i in epoch:
                x_data_train, y_data_train = np.asarray(epoch[i][0]), np.asarray(epoch[i][1])
                sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
                loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
                percent_error = 100*loss/variance
                print("  -  training global_step {0:4d} error: {1:8.4f} {2:8.2f}%".format(global_step, loss, percent_error))
                global_step += 1
            epoch_count+=1
        print('optimization finished!')
        print('\nstarting testing...')
        score = sess.run(cost, feed_dict={x: test_data, y: target_data_test})
        percent_error = 100*score/variance
        pred = sess.run(prediction, feed_dict={x: test_data})
        print('---- test score : {:.4f}, {:.4f}% ----'.format(score, percent_error))

if __name__ == '__main__':
    main()
