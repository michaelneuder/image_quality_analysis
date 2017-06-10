#!/usr/bin/env python3

# -----------------------------
# convolution to compare images
# -----------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
from scipy import signal as sig
from PIL import Image as im

def main():

    print("\nconvolution --- image evaluation\n")

    # ----------------------- data ------------------------ #
    # ----------------------------------------------------- #
    # original_images: 96x96 image w/ int in [0,255]        #
    # reconstructed_images: 96x96 image w/ float in [0,255] #
    # comparison_images: 96x96 image w/ float in [0,1)      #
    # ----------------------------------------------------- #

    original_images = np.loadtxt("data/orig_3pics.txt")
    reconstructed_images = np.loadtxt("data/recon_3pics.txt")
    comparison_images = np.loadtxt("data/ssim_3pics.txt")

    # data is now a 3 X 96 X 96 array (3 square 96px images)
    original_images = original_images.reshape(3,96,96)
    reconstructed_images = reconstructed_images.reshape(3,96,96)
    comparison_images = comparison_images.reshape(3,96,96)

    # these are copys of the data but with each entry being its own list
    # i made two copy because i have been doing stuff with the non-dimension version separately
    original_images_dim1 = original_images.reshape(3,96,96,1)
    reconstructed_images_dim1 = reconstructed_images.reshape(3,96,96,1)
    comparison_images_dim1 = comparison_images.reshape(3,96,96,1)

    # start of the tf stuff
    sess = tf.Session()
    width = 96
    height = 96

    # this placeholder will recieve the image data from outside tf and turn it into a tensor
    x_image = tf.placeholder(tf.float32, shape = [None, width, height, 1])

    # these are the variables that will be learned, initial values not too important
    filter_conv = tf.Variable(tf.truncated_normal([5,5,1,1]))
    bias_conv = tf.Variable(tf.constant(0.1))

    # the convolution operation, strides is how much it travels between each dot product.
    # ----------------------------------------------------------------------------------------#
    ## NOTE: this is actually dope of tensor flow. when we specify the padding as same, then  #
    ## it automagically chooses the right number of zeros to pad in order to give the output  #
    ## the same size as the input. so that is take care of for us. you can check this by      #
    ## changing the size of the filter. the output of the results.shape function will always  #
    ## be 96,96,3,1.                                                                          #
    # ----------------------------------------------------------------------------------------#
    convolution = tf.nn.conv2d(x_image, filter_conv, strides=[1,1,1,1], padding='SAME') + bias_conv

    # running the operation --- we run it on the original and the reconstructed
    init = tf.global_variables_initializer()
    sess.run(init)
    result_original = sess.run(convolution, feed_dict = {x_image: original_images_dim1})
    result_recon = sess.run(convolution, feed_dict = {x_image: reconstructed_images_dim1})

    # flattening out the images, because we arent using the square structure anymore
    ## this process is combining the original and reconstructed convolution into one array
    ## of length 18432 (96*96*2). this is to use the two images combined for our mlp training
    ## NOTE: i am sure there is a more efficient way to do this
    result_original = tf.reshape(result_original, [3, 9216])
    result_recon = tf.reshape(result_recon, [3, 9216])
    result_combined1 = tf.concat([result_original[0], result_recon[0]], 0)
    result_combined2 = tf.concat([result_original[1], result_recon[1]], 0)
    result_combined3 = tf.concat([result_original[2], result_recon[2]], 0)
    result_combined1 = tf.reshape(result_combined1, [1, 18432])
    result_combined2 = tf.reshape(result_combined2, [1, 18432])
    result_combined3 = tf.reshape(result_combined3, [1, 18432])
    result_total = tf.concat([result_combined1, result_combined2, result_combined3], 0)
    # print(result_total.shape)

    # this is the start of the MLP aspect of the network.
    ## x is the input from our combined result of the convolution
    ## y_ is the output, which is an array holding the resulting values
    x = tf.placeholder(tf.float32, shape=[None, 18432])
    y_ = tf.placeholder(tf.float32, shape=[None, 9612])

    # variables to be learned
    weights = tf.Variable(tf.zeros([18432, 9612], tf.float32))
    bias = tf.Variable(tf.zeros([9612], tf.float32))
    sess.run(tf.global_variables_initializer())

    # operations --- sigmoid normalizes the result
    # apply_weights_op = tf.matmul(x, weight)
    # add_bias_op = tf.add(apply_weights_op, bias)
    # activation_op = tf.nn.sigmoid(add_bias_op)

    y = tf.nn.sigmoid(tf.matmul(x, weights) +  bias)
    number_epochs = 1000
    learning_rate = .0001

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    
    y_1 = comparison_images
    y_1 = y_1.reshape(3,1,9216)


    # looking at images --- i just did this because i was curious was the images were.
    # if you want to see just uncomment the image_view.show() line
    # you can see the reconstruction by switching which one is commented out. pretty cool stuff
    image = np.asarray(original_images[1], dtype='uint8')
    # image = np.asarray(reconstructed_images[1], dtype='uint8')
    image_view = im.fromarray(image, 'L')
    # image_view.save("images/test.png")
    # image_view.show()

if __name__ == '__main__':
    main()
