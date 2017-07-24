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
    input_layer = 2
    first_layer = 70
    second_layer = 30
    third_layer = 15
    output_layer = 1
    number_images = 1

    print('generating random images ... ')
    rand_img_1 = np.random.random_sample((2,5,5))
    rand_img_2 = np.random.random_sample((2,5,5))
    difference = abs(rand_img_1 - rand_img_2)

    train_data_temp = []
    for i in range(2):
        for j in range(5):
            for k in range(5):
                train_data_temp.append([rand_img_1[i][j][k], rand_img_2[i][j][k]])
    train_data = np.reshape(np.asarray(train_data_temp), [2,5,5,2])
    for i in range(5):
        print(rand_img_1[0][0][i], rand_img_2[0][0][i], train_data[0][0][i])
    # for i in range(5):
    #     print(rand_img_1[0][0][i], rand_img_2[0][0][i], train_data[0][0][i])

    # train_data = np.dstack((rand_img_1, rand_img_2))

    # for i in range(96):
    #     print(rand_img_1[0][0][i], rand_img_2[0][0][i], train_data[0][0][i])
    # exit()
    # target_data = np.reshape(difference, [number_images,96,96,1])
    # target_data = np.reshape(sums, [1,96,96,1])

    exit()

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
    x = tf.placeholder(tf.float32, [None, 96, 96, 2])
    y = tf.placeholder(tf.float32, [None, 96, 96, 1])

    # paramaters
    learning_rate = .001
    epochs = 10000

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
            x_data_train, y_data_train = train_data, target_data
            sess.run(optimizer, feed_dict={x : x_data_train, y : y_data_train})
            loss = sess.run(cost, feed_dict={x : x_data_train, y : y_data_train})
            print("  -  training global_step {}. current error: {}. ".format(epoch_count, loss))
            epoch_count+=1
        print('optimization finished!')

        pred = np.asarray(sess.run(prediction, feed_dict={x : train_data}))
        target = np.asarray(target_data)
        print(pred.shape, target.shape)
        with open('post_training.csv', mode = 'w') as write_file:
            write_file.write('target, prediction\n')
            for i in range(96):
                for j in range(96):
                    write_file.write(str(float(target[0][i][j][0])) + ', ' + str(float(pred[0][i][j][0])) + '\n')
        write_file.close()
        with open('post_training_filters', mode='w') as write_file:
            write_file.write('weights 1\n')
            write_file.write(str(sess.run(weights['weights1'])))
            write_file.write('\nweights 2\n')
            write_file.write(str(sess.run(weights['weights2'])))
            write_file.write('\nweights output\n')
            write_file.write(str(sess.run(weights['weights_out'])))
            write_file.write('\nbias 1\n')
            write_file.write(str(sess.run(biases['bias1'])))
            write_file.write('\nbias 2\n')
            write_file.write(str(sess.run(biases['bias2'])))
            write_file.write('\nbias output\n')
            write_file.write(str(sess.run(biases['bias_out'])))
        write_file.close()
        rand_img_3 = np.random.random_sample((96,96))
        rand_img_4 = np.random.random_sample((96,96))
        sums = rand_img_3 + rand_img_4
        difference = abs(rand_img_3 - rand_img_4)
        train_data = np.reshape(np.dstack((rand_img_3, rand_img_4)), [1,96,96,2])
        # target_data = np.reshape(sums, [1,96,96,1])
        target_data = np.reshape(difference, [1,96,96,1])
        score = sess.run(cost, feed_dict={x: train_data, y: target_data})
        print('---- score : {} ----'.format(score))
        pred = sess.run(prediction, feed_dict={x: train_data})
        for i in range(96):
            # print(rand_img_3[0][i],rand_img_4[0][i],pred[0][0][i], sums[0][i])
            print(rand_img_3[0][i],rand_img_4[0][i],pred[0][0][i], difference[0][i])

if __name__ == '__main__':
    main()
