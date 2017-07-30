#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import time

def main():
    # data
    original_images_train = np.loadtxt('../../data/sample_data/orig_3pics.txt')
    original_images_train_re = np.reshape(original_images_train, [3,96,96,1])
    reconstructed_images_train = np.loadtxt('../../data/sample_data/recon_3pics.txt')
    reconstructed_images_train_re = np.reshape(reconstructed_images_train, [3,96,96,1])
    comparison_images_train = np.loadtxt('../../data/sample_data/comp_3pics.txt')
    original_images_test = np.loadtxt('../../data/sample_data/orig_3pics.txt')
    reconstructed_images_test = np.loadtxt('../../data/sample_data/recon_3pics.txt')
    comparison_images_test = np.loadtxt('../../data/sample_data/comp_3pics.txt')

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

    for i in range(3):
        for j in range(96):
            for k in range(96):
                print('-----------')
                print(input_combined_train[i][j][k][0] - original_images_train_re[i][j][k][0])
                print(input_combined_train[i][j][k][1] - reconstructed_images_train_re[i][j][k][0])
                print('-----------')

if __name__ == '__main__':
    main()
