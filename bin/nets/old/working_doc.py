#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf

# wrapper functions
def convolve_inner_layers(x, W, b):
    x = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def convolve_ouput_layer(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.sigmoid(x)

demo_orig = tf.Variable(tf.random_normal([96*96]))
demo_recon = tf.Variable(tf.random_normal([96*96]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# zipping data
demo_combined = []
for i in range(96*96):
    demo_combined.append([demo_orig[i], demo_recon[i]])

# layer 1 weights and bias
demo_weight = tf.Variable(tf.random_normal([5,5,2,10]))
demo_bias = tf.Variable(tf.random_normal([10]))

# layer 2 weights and bias
demo_weight2 = tf.Variable(tf.random_normal([5,5,10,20]))
demo_bias2 = tf.Variable(tf.random_normal([20]))

# layer 3 weights and bias
demo_weight3 = tf.Variable(tf.random_normal([5,5,20,30]))
demo_bias3 = tf.Variable(tf.random_normal([30]))

demo_weight_out = tf.Variable(tf.random_normal([5,5,30,1]))
demo_bias_out = tf.Variable(tf.random_normal([1]))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    demo_combined = tf.reshape(sess.run(demo_combined), shape= [-1,96,96,2])
    print('input data shape:', demo_combined.shape)

    conv1 = convolve_inner_layers(demo_combined, demo_weight, demo_bias)
    print('conv layer 1 output shape:', conv1.shape)

    conv2 = convolve_inner_layers(conv1, demo_weight2, demo_bias2)
    print('conv layer 2 output shape:', conv2.shape)

    conv3 = convolve_inner_layers(conv2, demo_weight3, demo_bias3)
    print('conv layer 3 output shape:', conv3.shape)

    output = convolve_ouput_layer(conv3, demo_weight_out, demo_bias_out)
    print('output layer shape:',output.shape)
