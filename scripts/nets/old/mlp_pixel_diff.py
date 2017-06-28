#!/usr/bin/env python3

# -----------------------------
# convolution to compare images
# -----------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)

def main():

    print("\nmulti-layer perceptron --- image evaluation\n")

    # ----------------------- data ------------------------ #
    # ----------------------------------------------------- #
    # original_images: 96x96 image w/ int in [0,255]        #
    # reconstructed_images: 96x96 image w/ float in [0,255] #
    # comparison_images: 96x96 image w/ float in [0,1)      #
    # ----------------------------------------------------- #

    # reading in data from files. original and recon are subtracted, lots of reshaping
    original_images_train = np.loadtxt("data/orig_train.txt")
    reconstructed_images_train = np.loadtxt("data/recon_train.txt")
    comparison_images_train = np.loadtxt("data/comp_train.txt")
    original_images_test = np.loadtxt("data/orig_test.txt")
    reconstructed_images_test = np.loadtxt("data/recon_test.txt")
    comparison_images_test = np.loadtxt("data/comp_test.txt")
    x_data_train = []
    y_data_train = []
    x_data_test = []
    y_data_test = []
    for i in range(original_images_train.shape[0]):
        for j in range(original_images_train.shape[1]):
            x_data_train.append(original_images_train[i][j] - reconstructed_images_train[i][j])
        y_data_train.append(comparison_images_train[i])
    for i in range(original_images_test.shape[0]):
        for j in range(original_images_test.shape[1]):
            x_data_test.append(original_images_test[i][j] - reconstructed_images_test[i][j])
        y_data_test.append(comparison_images_test[i])
    x_data_train = np.asarray(x_data_train)
    y_data_train = np.asarray(y_data_train)
    x_data_test = np.asarray(x_data_test)
    y_data_test = np.asarray(y_data_test)
    x_data_train = x_data_train.reshape(original_images_train.shape[0], 9216)
    y_data_train = y_data_train.reshape(original_images_train.shape[0], 9216)
    x_data_test = x_data_test.reshape(original_images_test.shape[0], 9216)
    y_data_test = y_data_test.reshape(original_images_test.shape[0], 9216)

    # start of the tf stuff
    sess = tf.Session()

    # running the operation --- we run it on the original and the reconstructed
    init = tf.global_variables_initializer()
    sess.run(init)

    # this is the start of the MLP aspect of the network.
    ## x is the input from our combined result of the convolution
    ## y_ is the output, which is an array holding the resulting values
    x = tf.placeholder(tf.float32, shape=[None, 9216])
    y_ = tf.placeholder(tf.float32, shape=[None, 9216])

    # variables to be learned
    weights = tf.Variable(tf.zeros([9216, 9216], tf.float32))
    bias = tf.Variable(tf.zeros([9216], tf.float32))
    sess.run(tf.global_variables_initializer())

    # some more parameters
    y = tf.nn.sigmoid(tf.matmul(x, weights) +  bias)
    learning_rate = .01

    # telling tf how to do the training
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # training
    train.run(session=sess, feed_dict={x: x_data_train, y_: y_data_train})

    # testing
    accuracy = tf.reduce_mean(tf.subtract(y, y_))
    acc = accuracy.eval(session=sess, feed_dict={x: x_data_train, y_:y_data_train})
    print(acc)

if __name__ == '__main__':
    main()
