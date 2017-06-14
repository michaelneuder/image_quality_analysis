#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf

# wrapper functions for convolutional layers
def convovle_inner_layers(x, W, b):
    x = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def convolve_ouput_layer(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.sigmoid(x)

# network model
def conv_net(x, weights, biases):
    conv1 = convovle_inner_layers(x, weights['weights1'], biases['bias1'])
    conv2 = convovle_inner_layers(conv1, weights['weights2'], biases['bias2'])
    conv3 = convovle_inner_layers(conv2, weights['weights3'], biases['bias3'])
    output = convolve_ouput_layer(conv3, weights['weights_out'], biases['bias_out'])
    return output

def get_batch(x, y):
    batch_indices = np.arange(x.shape[0])
    np.random.shuffle(batch_indices)
    x_batch = []
    y_batch = []
    for i in batch_indices[:50]:
        x_batch.append(x[i])
        y_batch.append(y[i])
    return [x_batch, y_batch]

def main():
    weights = {
        'weights1': tf.Variable(tf.random_normal([11,11,2,30])),
        'weights2': tf.Variable(tf.random_normal([11,11,30,20])),
        'weights3': tf.Variable(tf.random_normal([11,11,20,10])),
        'weights_out': tf.Variable(tf.random_normal([11,11,10,1]))
    }
    biases = {
        'bias1': tf.Variable(tf.random_normal([30])),
        'bias2': tf.Variable(tf.random_normal([20])),
        'bias3': tf.Variable(tf.random_normal([10])),
        'bias_out': tf.Variable(tf.random_normal([1]))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 96, 96, 2])
    y = tf.placeholder(tf.float32, [None, 96, 96, 1])

    # data
    original_images_train = np.loadtxt('../../data/sample_data/orig_500.txt')
    reconstructed_images_train = np.loadtxt('../../data/sample_data/recon_500.txt')
    comparison_images_train = np.loadtxt('../../data/sample_data/comp_500.txt')
    original_images_test = np.loadtxt('../../data/sample_data/orig_140.txt')
    reconstructed_images_test = np.loadtxt('../../data/sample_data/recon_140.txt')
    comparison_images_test = np.loadtxt('../../data/sample_data/comp_140.txt')

    # data
    # original_images_train = np.loadtxt('../../data/sample_data/orig_3pics.txt')
    # reconstructed_images_train = np.loadtxt('../../data/sample_data/recon_3pics.txt')
    # comparison_images_train = np.loadtxt('../../data/sample_data/comp_3pics.txt')
    # original_images_test = np.loadtxt('../../data/sample_data/orig_3pics.txt')
    # reconstructed_images_test = np.loadtxt('../../data/sample_data/recon_3pics.txt')
    # comparison_images_test = np.loadtxt('../../data/sample_data/comp_3pics.txt')

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
    learning_rate = .001
    training_iterations = 10

    # model
    prediction = conv_net(x, weights, biases)

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # evaluation --- same as the cost function
    error = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(weights['weights1'][0][0]))
        step = 1
        print("--------------------------------------------------")
        print("starting training ... ")
        while step < training_iterations:
            batch = get_batch(input_combined_train, comparison_images_train)
            x_data_train , y_data_train = np.asarray(batch[0]), np.asarray(batch[1])
            # x_data_train, y_data_train = [input_combined_train[0]], [comparison_images_train[0]]
            sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
            loss = sess.run(error, feed_dict={x : x_data_train, y : y_data_train})
            print("training step {}. current error: {}. ".format(step, loss))
            step += 1
        print("optimization finished!")
        print("--------------------------------------------------")
        print("testing accuracy")
        x_data_test, y_data_test = input_combined_test.reshape([test_size,96,96,2]), comparison_images_test
        # x_data_test, y_data_test = [input_combined_test[0]], [comparison_images_test[0]]
        final_error = sess.run(error, feed_dict={x: x_data_test, y: y_data_test})
        print("the average pixel difference on the test set is {}.".format(final_error))
        print(sess.run(weights['weights1'][0][0]))

if __name__ == '__main__':
    main()
