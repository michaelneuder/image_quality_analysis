#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def get_variance(training_target):
    '''
    returns variance of the target data. used in normalizing the error.
    '''
    all_pixels = training_target.flatten()
    return all_pixels.var()

def normalize_input(train_data, test_data):
    '''
    normailizing input across each pixel an each channel (i.e. normalize for each input to network).
    '''
    mean, std_dev = np.mean(train_data, axis=0), np.std(train_data, axis=0)
    return (train_data - mean) / std_dev, (test_data - mean) / std_dev

def get_epoch(x, y, n):
    '''
    splits entire data set into an epoch with minibatch of size n. returns a dict with key being the
    minibatch number and the value being a length 2 list with the features in first index and
    targets in the second.
    '''
    input_size = x.shape[0]
    number_batches = input_size // n
    extra_examples = input_size % n
    batches = {}
    batch_indices = np.arange(input_size)
    np.random.shuffle(batch_indices)
    for i in range(number_batches):
        temp_indices = batch_indices[n*i:n*(i+1)]
        temp_x = []
        temp_y = []
        for j in temp_indices:
            temp_x.append(x[j])
            temp_y.append(y[j])
        batches[i] = [np.asarray(temp_x), np.asarray(temp_y)]
    if extra_examples != 0:
        extra_indices = batch_indices[input_size-extra_examples:input_size]
        temp_x = []
        temp_y = []
        for k in extra_indices:
            temp_x.append(x[k])
            temp_y.append(y[k])
        batches[i+1] = [np.asarray(temp_x), np.asarray(temp_y)]
    return batches

def calculate_ssim_patch(window_orig, window_recon):
    '''
    calculates the ssim value of two 11x11 patches.
    '''
    k_1, k_2, L = 0.01, 0.03, 255
    if window_orig.shape != (11,11) or window_recon.shape != (11,11):
        raise ValueError('please check window size for SSIM calculation!')
    orig_data, recon_data = window_orig.flatten(), window_recon.flatten()
    mean_x, mean_y = np.mean(orig_data), np.mean(recon_data)
    var_x, var_y = np.var(recon_data), np.var(orig_data)
    covar = np.cov(orig_data, recon_data)[0][1]
    c_1, c_2 = (L*k_2)**2, (L*k_1)**2
    num = (2*mean_x*mean_y+c_1)*(2*covar+c_2)
    den = (mean_x**2+mean_y**2+c_1)*(var_x+var_y+c_2)
    return num/den

def calculate_ssim_image(image_orig, image_recon):
    '''
    returns ssim map for entire image.
    '''
    ssim_res = []
    filter_dim = 11; image_dim = image_orig.shape[0];
    number_windows = image_dim - filter_dim + 1
    for i in range(number_windows):
        for j in range(number_windows):
            orig_window = image_orig[i:i+11, j:j+11]
            recon_window = image_recon[i:i+11, j:j+11]
            temp = calculate_ssim_patch(orig_window, recon_window)
            ssim_res.append(temp)
    return np.asarray(ssim_res)
