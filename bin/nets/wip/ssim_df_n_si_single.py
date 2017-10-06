#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time
import pandas as pd

def convolve_inner_layers(x, W, b):
    y = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='VALID')
    y = tf.nn.bias_add(y, b)
    return tf.nn.tanh(y)

def convolve_ouput_layer(x, W, b):
    y = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='VALID')
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

def calculate_ssim(window_orig, window_recon):
    k_1 = 0.01
    k_2 = 0.03
    L = 255
    if window_orig.shape != (11,11) or window_recon.shape != (11,11):
        raise ValueError('please check window size for SSIM calculation!')
    orig_data = window_orig.flatten()
    recon_data = window_recon.flatten()
    mean_x = np.mean(orig_data)
    mean_y = np.mean(recon_data)
    var_x = np.var(orig_data)
    var_y = np.var(recon_data)
    covar = np.cov(orig_data, recon_data)[0][1]
    c_1 = (L*k_1)**2
    c_2 = (L*k_2)**2
    num = (2*mean_x*mean_y+c_1)*(2*covar+c_2)
    den = (mean_x**2+mean_y**2+c_1)*(var_x+var_y+c_2)
    return num/den

def main():
    # parameters
    filter_dim = 11
    filter_dim2 = 1
    batch_size = 200
    image_dim = 96
    input_layer = 4
    first_layer = 50
    second_layer = 25
    third_layer = 10
    output_layer = 1
    learning_rate = .01
    epochs = 400

    # seeding for debug purposes --- dont forget to remove
    # SEED = 12345
    # np.random.seed(SEED)
    # tf.set_random_seed(SEED)

    print('loading image files ... ')
    # train/test images
    orig_500 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/orig_500.txt', header=None, delim_whitespace = True)
    recon_500 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/recon_500.txt', header=None, delim_whitespace = True)
    SSIM_500 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/SSIM_500.txt', header=None, delim_whitespace = True)
    orig_140 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/orig_140.txt', header=None, delim_whitespace = True)
    recon_140 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/recon_140.txt', header=None, delim_whitespace = True)
    SSIM_140 = pd.read_csv('https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/SSIM_140.txt', header=None, delim_whitespace = True)

    # getting 4 input channels for train and test
    original_images_train = orig_500.values
    original_images_train_sq = orig_500.values**2
    reconstructed_images_train = recon_500.values
    reconstructed_images_train_sq = recon_500.values**2

    original_images_test = orig_140.values
    original_images_test_sq = orig_140.values**2
    reconstructed_images_test = recon_140.values
    reconstructed_images_test_sq = recon_140.values**2

    # stack inputs
    training_input = np.dstack((original_images_train, reconstructed_images_train, original_images_train_sq, reconstructed_images_train_sq))
    testing_input = np.dstack((original_images_test, reconstructed_images_test, original_images_test_sq, reconstructed_images_test_sq))

    # normalize inputs
    training_input_normalized, testing_input_normalized = normalize_input(training_input, testing_input)

    # target values
    comparison_images_train = SSIM_500.values
    comparison_images_test = SSIM_140.values

    # get size of training and testing set
    train_size = original_images_train.shape[0]
    test_size = original_images_test.shape[0]

    # reshaping the result data to --- (num pics), 96, 96, 1
    target_data_train = np.reshape(comparison_images_train, [train_size, image_dim, image_dim, output_layer])
    target_data_test = np.reshape(comparison_images_test, [test_size, image_dim, image_dim, output_layer])

    # reshaping
    train_data = np.reshape(training_input_normalized, [train_size,image_dim,image_dim,input_layer])
    test_data =  np.reshape(testing_input_normalized, [test_size,image_dim,image_dim,input_layer])

    image_dim = 11
    single_train_data, single_test_data = [], []

    for i in range(train_data.shape[0]):
        for j in range(11):
            for k in range(11):
                single_train_data.append(train_data[i,j,k])
                if i < 140:
                    single_test_data.append(test_data[i,j,k])

    single_train_data = np.reshape(np.asarray(single_train_data), (train_data.shape[0], 11, 11, 4))
    single_test_data = np.reshape(np.asarray(single_test_data), (test_data.shape[0], 11, 11, 4))

    ssim, ssim1 = [], []
    for i in range(single_train_data.shape[0]):
        ssim.append(calculate_ssim(single_train_data[i][...,0], single_train_data[i][...,1]))
        if i < 140:
            ssim1.append(calculate_ssim(single_test_data[i][...,0], single_test_data[i][...,1]))

    ssim = np.reshape(np.asarray(ssim), (single_train_data.shape[0],1))
    ssim1 = np.reshape(np.asarray(ssim1), (single_test_data.shape[0],1))

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
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, input_layer])
    y = tf.placeholder(tf.float32, [None, output_layer])

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
            epoch = get_epoch(single_train_data, ssim, batch_size)
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
        score = sess.run(cost, feed_dict={x: single_test_data, y: ssim1})
        percent_error = 100*score/variance
        print('---- test score : {:.4f}, {:.4f}% ----'.format(score, percent_error))

if __name__ == '__main__':
    main()
