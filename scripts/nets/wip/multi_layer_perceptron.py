#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time
from PIL import Image as im

def multilayer_perceptron(x, weights, biases):
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer

def main():
    print('multi-layer perceptron')
    print('------------------------------------------------------')
    print("initializing variables ...")

    input_size = 96*96*2
    output_size = 96*96

    weights = {
        'out': tf.Variable((1/(input_size))*tf.random_normal([input_size, output_size]))
    }
    biases = {
        'out': tf.Variable((1/(input_size))*tf.random_normal([output_size]))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, output_size])

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

    combined_data = np.reshape(np.dstack((original_images_train, reconstructed_images_train)), [train_size, input_size])

    # print(test[0].shape)
    # orig = []
    # recon = []
    # for i in range(96):
    #     for j in range(96):
    #         orig.append(test[49][i][j][0])
    #         recon.append(test[49][i][j][1])
    #
    # image1 = np.reshape(np.asarray(orig, dtype='uint8'), [96,96])
    # image2 = np.reshape(np.asarray(recon, dtype='uint8'), [96,96])
    # image_view = im.fromarray(image1, 'L')
    # image_view.show()
    # image_view = im.fromarray(image2, 'L')
    # image_view.show()

    # paramaters
    learning_rate = .000001
    epochs = 150

    # model
    prediction = multilayer_perceptron(x, weights, biases)

    # saving state
    saver = tf.train.Saver()

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
            x_data_train, y_data_train = np.asarray([combined_data[0]]), np.asarray([comparison_images_train[0]])
            print(x_data_train.shape, y_data_train.shape)
            sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
            loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
            print("  -  training global_step {}. current error: {}. ".format(global_step, loss))
            global_step+=1
            print('epoch {} completed in {} seconds. current error = {}'.format(epoch_count, time.time()-epoch_time, loss))
            print('-------------------------------------------------------')
            epoch_count+=1
        print('optimization finished!')

        test = np.asarray(sess.run(prediction, feed_dict={x : x_data_train}))
        test1 = np.asarray([comparison_images_train[0]])
        print(test.shape, test1.shape)
        with open('bias_learned.txt', mode = 'w') as write_file:
            write_file.write('prediction:\n')
            write_file.write(str(test)+'\n')
            write_file.write('target:\n')
            write_file.write(str(test1))
        write_file.close()

if __name__ == '__main__':
    main()
