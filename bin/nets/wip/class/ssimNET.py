#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ssimNET(object):
    '''
    an instance of the ssim convolutional neural network.
    '''
    def __init__(self, data_path='', layer_sizes=[4,100,50,32,10,1]):
        '''
        loads data into class variables and initializes network parameters
        '''
        self.epochs=5000; self.learning_rate=0.001; self.batch_size=4;
        self.filter_dims = [11,1]; self.image_dim=96; self.result_dim=86;
        self.data_path = data_path; self.layer_sizes = layer_sizes;
        self.train_features=[]; self.train_target=[]; self.test_features=[]; self.test_target=[];

    def load_data(self, local=False):
        '''
        loads data and pepares it for network.
        '''
        if local and (self.data_path == ''):
            raise ValueError('please specify a data path in the constructor')
        if local:
            data_path = self.data_path
        else:
            data_path = 'https://raw.githubusercontent.com/michaelneuder/image_quality_analysis/master/data/sample_data/'
        image_dim, result_dim = 96, 86
        input_layer, output_layer = 4, 1
        input_layer, first_layer, second_layer, third_layer, fourth_layer, output_layer = 4, 100, 50, 25, 10, 1
        filter_dim, filter_dim2 = 11, 1

        # train data --- 500 images, 96x96 pixels
        orig_500 = pd.read_csv('{}orig_500.txt'.format(data_path), header=None, delim_whitespace = True)
        recon_500 = pd.read_csv('{}recon_500.txt'.format(data_path), header=None, delim_whitespace = True)

        # test data --- 140 images, 96x96 pixels
        orig_140 = pd.read_csv('{}orig_140.txt'.format(data_path), header=None, delim_whitespace = True)
        recon_140 = pd.read_csv('{}recon_140.txt'.format(data_path), header=None, delim_whitespace = True)

        # targets
        ssim_500 = pd.read_csv('{}ssim_500_nogauss.csv'.format(data_path), header=None)
        ssim_140 = pd.read_csv('{}ssim_140_nogauss.csv'.format(data_path), header=None)

        # getting 4 input channels for train and test --- (orig, recon, orig squared, recon squared)
        original_images_train = orig_500.values
        original_images_train_sq = orig_500.values**2
        reconstructed_images_train = recon_500.values
        reconstructed_images_train_sq = recon_500.values**2

        original_images_test = orig_140.values
        original_images_test_sq = orig_140.values**2
        reconstructed_images_test = recon_140.values
        reconstructed_images_test_sq = recon_140.values**2

        # stack inputs
        training_input = np.dstack((original_images_train, reconstructed_images_train, original_images_train_sq, reconstructed_images_train_sq))
        testing_input = np.dstack((original_images_test, reconstructed_images_test, original_images_test_sq, reconstructed_images_test_sq))

        # normalize inputs
        training_input_normalized, testing_input_normalized = normalize_input(training_input, testing_input)

        # target values
        training_target = ssim_500.values
        testing_target = ssim_140.values

        # get size of training and testing set
        train_size = original_images_train.shape[0]
        test_size = original_images_test.shape[0]

        # reshaping features to (num images, 96x96, 4 channels)
        train_features = np.reshape(training_input_normalized, [train_size,image_dim,image_dim,input_layer])
        test_features =  np.reshape(testing_input_normalized, [test_size,image_dim,image_dim,input_layer])

        # reshaping target to --- (num images, 86x86, 1)
        train_target = np.reshape(training_target, [train_size, result_dim, result_dim, output_layer])
        test_target = np.reshape(testing_target, [test_size, result_dim, result_dim, output_layer])

        self.train_features = train_features; self.train_target = train_target;
        self.test_features = test_features; self.test_target = test_target;
        return

    def normalize_input(self):
