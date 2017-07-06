#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time

def conv_net(x, W, b):
    y = tf.nn.conv2d(x, W['weights_out'], strides=[1,1,1,1], padding='SAME')
    y = tf.nn.bias_add(y, b['bias_out'])
    return y

def main():
    print('generating random images ... ')
    rand_img_1 = np.random.random_sample((96,96))
    rand_img_2 = np.random.random_sample((96,96))
    difference = abs(rand_img_1 - rand_img_2)

    combined_image_data = np.reshape(np.dstack((rand_img_1, rand_img_2)), [1,96,96,2])
    difference_image_data = np.reshape(difference, [1,96,96,1])

    weights = {
        'weights_out': tf.Variable((1/(11*11))*tf.random_normal([11,11,2,1]))
    }
    biases = {
        'bias_out': tf.Variable((1/(11*11))*tf.random_normal([1]))
    }

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 96, 96, 2])
    y = tf.placeholder(tf.float32, [None, 96, 96, 1])

    # paramaters
    learning_rate = .01
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
        global_step = 0
        epoch_count = 0
        start_time = time.time()
        print("starting training ... ")
        while epoch_count < epochs:
            x_data_train, y_data_train = combined_image_data, difference_image_data
            sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
            loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
            print("  -  training global_step {}. current error: {}. ".format(global_step, loss))
            global_step+=1
            epoch_count+=1
        print('optimization finished!')
        #
        # prediction = np.asarray(sess.run(prediction, feed_dict={x : [combined_data_train[0]]}))
        # target = np.asarray([comparison_images_test[0]])
        # print(prediction.shape, target.shape)
        # with open('post_training.csv', mode = 'w') as write_file:
        #     write_file.write('target, prediction\n')
        #     for i in range(96):
        #         for j in range(96):
        #             write_file.write(str(float(target[0][i][j][0])) + ', ' + str(float(prediction[0][i][j][0])) + '\n')
        # write_file.close()



if __name__ == '__main__':
    main()
