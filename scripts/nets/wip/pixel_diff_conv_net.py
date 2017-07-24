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
    return tf.nn.relu(y)
    # return y

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

def main():
    # parameters
    filter_dim = 1
    number_images = 10
    image_dim = 20
    input_layer = 2
    first_layer = 70
    second_layer = 30
    third_layer = 15
    output_layer = 1

    print('generating random images ... ')
    # train images
    rand_img_train_1 = np.random.random_sample((number_images,image_dim**2))
    rand_img_train_2 = np.random.random_sample((number_images,image_dim**2))
    difference_train = abs(rand_img_train_1 - rand_img_train_2)

    # test image
    rand_img_test_1 = np.random.random_sample((1,image_dim**2))
    rand_img_test_2 = np.random.random_sample((1,image_dim**2))
    difference_test = abs(rand_img_test_1 - rand_img_test_2)

    # stacking & reshaping images
    train_data = np.reshape(np.dstack((rand_img_train_1, rand_img_train_2)), [number_images,image_dim,image_dim,2])
    test_data = np.reshape(np.dstack((rand_img_test_1, rand_img_test_2)), [1,image_dim,image_dim,2])
    target_data_train = np.reshape(difference_train, [number_images,image_dim,image_dim,1])
    target_data_test = np.reshape(difference_test, [1,image_dim,image_dim,1])

    weights = {
        'weights1': tf.Variable((1/(filter_dim*filter_dim*input_layer))*tf.random_normal([filter_dim,filter_dim,input_layer,first_layer])),
        'weights2': tf.Variable((1/(filter_dim*filter_dim*first_layer))*tf.random_normal([filter_dim,filter_dim,first_layer,second_layer])),
        'weights3': tf.Variable((1/(filter_dim*filter_dim*second_layer))*tf.random_normal([filter_dim,filter_dim,second_layer,third_layer])),
        'weights_out': tf.Variable((1/(filter_dim*filter_dim*third_layer))*tf.random_normal([filter_dim,filter_dim,third_layer,output_layer]))
    }
    biases = {
        'bias1': tf.Variable((1/(filter_dim*filter_dim*first_layer))*tf.random_normal([first_layer])),
        'bias2': tf.Variable((1/(filter_dim*filter_dim*second_layer))*tf.random_normal([second_layer])),
        'bias3': tf.Variable((1/(filter_dim*filter_dim*third_layer))*tf.random_normal([third_layer])),
        'bias_out': tf.Variable((1/(filter_dim*filter_dim*output_layer))*tf.random_normal([output_layer]))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, 2])
    y = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])

    # paramaters
    learning_rate = .001
    epochs = 1000

    # model
    prediction = conv_net(x, weights, biases)

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        epoch_count = 0
        start_time = time.time()
        print("starting training ... ")
        while epoch_count < epochs:
            x_data_train, y_data_train = train_data, target_data_train
            sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
            loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
            print("  -  training global_step {}. current error: {}. ".format(epoch_count, loss))
            epoch_count+=1
        print('optimization finished!')
        print('\nstarting testing...')
        score = sess.run(cost, feed_dict={x: test_data, y: target_data_test})
        pred = sess.run(prediction, feed_dict={x: test_data})
        for i in range(image_dim):
            print(rand_img_test_1[0][i],rand_img_test_2[0][i], pred[0][0][i], difference_test[0][i])
        print('---- score : {} ----'.format(score))

if __name__ == '__main__':
    main()
