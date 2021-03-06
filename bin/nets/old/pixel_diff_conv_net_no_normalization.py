#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time

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
    output = convolve_ouput_layer(conv3, W['weights_out'], b['bias_out'])
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

def main():
    # parameters
    filter_dim = 11
    filter_dim2 = 1
    number_images = 100
    batch_size = 4
    image_dim = 96
    input_layer = 2
    first_layer = 50
    second_layer = 25
    third_layer = 10
    output_layer = 1
    initializer_scale = 1.0

    # seeding for debug purposes --- dont forget to remove
    SEED = 12345
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    print('generating random images ... ')
    # train images
    rand_img_train_1 = np.random.random_sample((number_images,image_dim**2))
    rand_img_train_2 = np.random.random_sample((number_images,image_dim**2))
    difference_train = abs(rand_img_train_1 - rand_img_train_2)

    # test image
    rand_img_test_1 = np.random.random_sample((number_images,image_dim**2))
    rand_img_test_2 = np.random.random_sample((number_images,image_dim**2))
    difference_test = abs(rand_img_test_1 - rand_img_test_2)

    # stacking & reshaping images
    train_data = np.reshape(np.dstack((rand_img_train_1, rand_img_train_2)), [number_images,image_dim,image_dim,2])
    test_data = np.reshape(np.dstack((rand_img_test_1, rand_img_test_2)), [number_images,image_dim,image_dim,2])
    target_data_train = np.reshape(difference_train, [number_images,image_dim,image_dim,1])
    target_data_test = np.reshape(difference_test, [number_images,image_dim,image_dim,1])

    # initializing variables --- fan in
    weights = {
        'weights1': tf.Variable(tf.random_normal([filter_dim,filter_dim,input_layer,first_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*input_layer)))),
        'weights2': tf.Variable(tf.random_normal([filter_dim2,filter_dim2,first_layer,second_layer],stddev=(1.0/(initializer_scale*filter_dim2*filter_dim2*first_layer)))),
        'weights3': tf.Variable(tf.random_normal([filter_dim2,filter_dim2,second_layer,third_layer],stddev=(1.0/(initializer_scale*filter_dim2*filter_dim2*second_layer)))),
        'weights_out': tf.Variable(tf.random_normal([filter_dim2,filter_dim2,third_layer,output_layer],stddev=(1.0/(initializer_scale*filter_dim2*filter_dim2*third_layer))))
    }
    biases = {
        'bias1': tf.Variable(tf.random_normal([first_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*input_layer)))),
        'bias2': tf.Variable(tf.random_normal([second_layer],stddev=(1.0/(initializer_scale*filter_dim2*filter_dim2*first_layer)))),
        'bias3': tf.Variable(tf.random_normal([third_layer],stddev=(1.0/(initializer_scale*filter_dim2*filter_dim2*second_layer)))),
        'bias_out': tf.Variable(tf.random_normal([output_layer],stddev=(1.0/(initializer_scale*filter_dim2*filter_dim2*third_layer))))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, 2])
    y = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])

    # paramaters
    learning_rate = .01
    epochs = 1000

    # model
    prediction = conv_net(x, weights, biases)

    # get variance to normalize error terms during training
    variance = get_variance(difference_train)

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
            epoch_time = time.time()
            print('--------------------------------------------------------------')
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
            print('epoch finished in {0:5.1f} seconds. total time training: {1:5.1f} seconds'.format(abs(epoch_time-time.time()), abs(start_time-time.time())))
        print('optimization finished!')
        print('\nstarting testing...')
        score = sess.run(cost, feed_dict={x: test_data, y: target_data_test})
        pred = sess.run(prediction, feed_dict={x: test_data})
        for i in range(image_dim):
            print(rand_img_test_1[0][i],rand_img_test_2[0][i], pred[0][0][i], difference_test[0][i])
        print('---- score : {} ----'.format(score))

if __name__ == '__main__':
    main()
