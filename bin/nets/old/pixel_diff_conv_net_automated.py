#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time

# seeding for debug purposes --- dont forget to remove
SEED = 12345
np.random.seed(SEED)
tf.set_random_seed(SEED)

def convolve_inner_layers(x, W, b):
    y = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return tf.nn.relu(y)

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

def run_training(image_dim_, initializer_scale_, learning_rate_):
    # parameters
    filter_dim = 11
    number_images = 100
    image_dim = image_dim_
    input_layer = 2
    first_layer = 50
    second_layer = 25
    third_layer = 10
    output_layer = 1
    initializer_scale = initializer_scale_

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
        'weights2': tf.Variable(tf.random_normal([filter_dim,filter_dim,first_layer,second_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*first_layer)))),
        'weights3': tf.Variable(tf.random_normal([filter_dim,filter_dim,second_layer,third_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*second_layer)))),
        'weights_out': tf.Variable(tf.random_normal([filter_dim,filter_dim,third_layer,output_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*third_layer))))
    }
    biases = {
        'bias1': tf.Variable(tf.random_normal([first_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*input_layer)))),
        'bias2': tf.Variable(tf.random_normal([second_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*first_layer)))),
        'bias3': tf.Variable(tf.random_normal([third_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*second_layer)))),
        'bias_out': tf.Variable(tf.random_normal([output_layer],stddev=(1.0/(initializer_scale*filter_dim*filter_dim*third_layer))))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, 2])
    y = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])

    # paramaters
    learning_rate = learning_rate_
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
        print("starting training with paramaers: (im_dim={}, init_scale={}, lr={})".format(image_dim, initializer_scale, learning_rate))
        while epoch_count < epochs:
            x_data_train, y_data_train = train_data, target_data_train
            sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
            loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
            epoch_count+=1
        print('    optimization finished!')
        score = sess.run(cost, feed_dict={x: test_data, y: target_data_test})
        print('    score : {} '.format(score))
    return (image_dim, initializer_scale, learning_rate), (loss, score)

def main():
    results = {}
    image_dims = [1,2,3,4,5]
    init_scales = [.01, .1, 1.0, 10.0]
    learning_rates = [.1, .01, .001]
    for dim in image_dims:
        for scale in init_scales:
            for learning_rate in learning_rates:
                setting, result = run_training(dim, scale, learning_rate)
                results[setting] = result
    with open('results.txt', mode='w') as write_file:
        for setting in results:
            write_file.write(str(setting)+','+str(results[setting])+'\n')
    write_file.close()

if __name__ == '__main__':
    main()
