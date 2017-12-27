#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib as mpl
import pandas as pd
import numpy as np
mpl.use('Agg')
import time
import matplotlib.pyplot as plt

def main():
    print('welcome to ssim net.')

    # parameters
    filter_dim, filter_dim2 = 11, 1
    batch_size = 4
    image_dim, result_dim = 96, 86
    input_layer, first_layer, second_layer, third_layer, fourth_layer, output_layer = 4, 10, 5, 2, 1, 1
    learning_rate = .001
    epochs = 5000

    # initializing filters, this is what we are trying to learn --- fan in
    scaling_factor = 0.1
    initializer = tf.contrib.layers.xavier_initializer()
    weights = {
        'weights1': tf.get_variable('weights1', [filter_dim,filter_dim,input_layer,first_layer], initializer=initializer),
        'weights2': tf.get_variable('weights2', [filter_dim2,filter_dim2,first_layer,second_layer], initializer=initializer),
        'weights3': tf.get_variable('weights3', [filter_dim2,filter_dim2,second_layer,third_layer], initializer=initializer),
        'weights4': tf.get_variable('weights4', [filter_dim2,filter_dim2,third_layer,fourth_layer], initializer=initializer),
        'weights_out': tf.get_variable('weights_out', [filter_dim2,filter_dim2,fourth_layer+third_layer+second_layer+first_layer,output_layer], initializer=initializer)
    }
    biases = {
        'bias1': tf.get_variable('bias1', [first_layer], initializer=initializer),
        'bias2': tf.get_variable('bias2', [second_layer], initializer=initializer),
        'bias3': tf.get_variable('bias3', [third_layer], initializer=initializer),
        'bias4': tf.get_variable('bias4', [fourth_layer], initializer=initializer),
        'bias_out': tf.get_variable('bias_out', [output_layer], initializer=initializer)
    }

    init = tf.global_variables_initializer()
    print('variables initialized ...')

    # tensorflow session & training
    with tf.Session() as sess:
        sess.run(init)
        for mat in weights:
            temp = np.asarray(sess.run(weights[mat]))
            orig_shape = temp.shape
            temp_flat = temp.flatten()
            np.savetxt('weights/{}.txt'.format(mat), temp_flat)
            temp_test = np.loadtxt('weights/{}.txt'.format(mat)).reshape(orig_shape)
            print(np.mean(temp - temp_test))
        for mat in biases:
            temp = np.asarray(sess.run(biases[mat]))
            orig_shape = temp.shape
            temp_flat = temp.flatten()
            np.savetxt('weights/{}.txt'.format(mat), temp_flat)
            temp_test = np.loadtxt('weights/{}.txt'.format(mat)).reshape(orig_shape)
            print(np.mean(temp - temp_test))


    print('training finished.')

if __name__ == '__main__':
    main()
