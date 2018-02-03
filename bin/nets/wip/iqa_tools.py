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
    input: entire training set.
    returns: the varince.
    '''
    all_pixels = training_target.flatten()
    return all_pixels.var()

def normalize_input(train_data, test_data):
    '''
    input: two image sets (1 training, 1 test)
    returns: two image sets of same dimension but standardized.
    '''
    mean, std_dev = np.mean(train_data, axis=0), np.std(train_data, axis=0)
    return (train_data - mean) / std_dev, (test_data - mean) / std_dev

def get_epoch(x, y, n):
    '''
    input: set of images, set of targets, size of batch
    returns: dict with key being the minibatch number and the value being a length 2 list with
    the features in first index and targets in the second.
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

def calculate_ssim(window_orig, window_recon):
    '''
    input: two 11x11 windows of orig and recon image.
    returns: ssim score (single value) for the patches
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
    input: orig and recon image
    returns: ssim score pixelwise with no zero-padding.
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

def calculate_contrast(window_orig, window_recon):
    '''
    input: orig and recon patch of image.
    returns: single contrast score for patches
    '''
    k_2, L = 0.03, 255
    c_2 = (L*k_2)**2

    orig_data, recon_data = window_orig.flatten(), window_recon.flatten()
    var_x, var_y = np.var(recon_data), np.var(orig_data)

    num = 2*np.sqrt(var_x)*np.sqrt(var_y) + c_2
    den = var_x + var_y + c_2

    return num/den

def calculate_structure(window_orig, window_recon):
    '''
    input: orig and recon patch of image.
    returns: single structure score for patches
    '''
    k_2, L = 0.03, 255
    c_2 = (L*k_2)**2
    c_3 = c_2 / 2

    orig_data, recon_data = window_orig.flatten(), window_recon.flatten()
    std_x, std_y = np.std(recon_data), np.std(orig_data)
    covar = np.cov(orig_data, recon_data)[0][1]

    num = covar + c_3
    den = std_x * std_y + c_3

    return num/den

def calculate_contrast_image(orig_im, recon_im):
    '''
    input: orig and recon image.
    returns: contrast scores pixelwise for the image.
    '''
    contrast_res = []
    number_windows = orig_im.shape[0] - filter_dim + 1
    for i in range(number_windows):
        for j in range(number_windows):
            orig_window = orig_im[i:i+11, j:j+11]
            recon_window = recon_im[i:i+11, j:j+11]
            temp = calculate_contrast(orig_window, recon_window)
            contrast_res.append(temp)
    return np.reshape(contrast_res, (number_windows, number_windows))

def calculate_structure_image(orig_im, recon_im):
    '''
    input: orig and recon image.
    returns: structure pixelwise.
    '''
    structure_res = []
    number_windows = orig_im.shape[0] - filter_dim + 1
    for i in range(number_windows):
        for j in range(number_windows):
            orig_window = orig_im[i:i+11, j:j+11]
            recon_window = recon_im[i:i+11, j:j+11]
            temp = calculate_structure(orig_window, recon_window)
            structure_res.append(temp)
    return np.reshape(structure_res, (number_windows, number_windows))

def down_sample(orig_im, recon_im, pool_size):
    '''
    input: orig, recon, size of pool
    return: a tuple of original and reconstructed images after down sampling
    '''
    reduce_im_orig, reduce_im_recon = [], []
    number_pools = int(orig_im.shape[0] / pool_size)
    for i in range(number_pools):
        for j in range(number_pools):
            orig_pool = orig_im[i*pool_size:i*pool_size+pool_size, j*pool_size:j*pool_size+pool_size]
            recon_pool = recon_im[i*pool_size:i*pool_size+pool_size, j*pool_size:j*pool_size+pool_size]
            temp_orig, temp_recon = np.mean(orig_pool), np.mean(recon_pool)
            reduce_im_orig.append(temp_orig)
            reduce_im_recon.append(temp_recon)

    return np.reshape(reduce_im_orig, (number_pools,number_pools)), np.reshape(reduce_im_recon, (number_pools,number_pools))

def calculate_luminance(window_orig, window_recon):
    '''
    input: patch of recon and orig image
    returns: luminance score for the patches
    '''
    k_1, L = 0.01, 255
    c_1 = (L*k_1)**2

    orig_data, recon_data = window_orig.flatten(), window_recon.flatten()
    mean_x, mean_y = np.mean(recon_data), np.mean(orig_data)

    num = 2*mean_x*mean_y + c_1
    den = np.square(mean_x)+ np.square(mean_y) + c_1

    return num/den

def calculate_luminance_image(orig_im, recon_im):
    '''
    input: orig and recon images
    returns: pixelwise luminance score with no zero padding.
    '''
    luminance_res = []
    number_windows = orig_im.shape[0] - filter_dim + 1
    for i in range(number_windows):
        for j in range(number_windows):
            orig_window = orig_im[i:i+11, j:j+11]
            recon_window = recon_im[i:i+11, j:j+11]
            temp = calculate_luminance(orig_window, recon_window)
            luminance_res.append(temp)
    return np.reshape(luminance_res, (number_windows, number_windows))

def calculate_msssim_image(orig, recon):
    '''
    input: orig and recon images
    returns: single msssim value for pair of images
    '''
    contrast1, structure1 = calculate_contrast_image(orig, recon), calculate_structure_image(orig, recon)
    orig_ds1, recon_ds1 = down_sample(orig, recon)
    contrast2, structure2 = calculate_contrast_image(orig_ds1, recon_ds1), calculate_structure_image(orig_ds1, recon_ds1)
    orig_ds2, recon_ds2 = down_sample(orig_ds1, recon_ds1)
    contrast3, structure3 = calculate_contrast_image(orig_ds2, recon_ds2), calculate_structure_image(orig_ds2, recon_ds2)
    luminance = calculate_luminance_image(orig_ds2, recon_ds2)
    return contrast1*contrast2*contrast3*structure1*structure2*structure3*luminance
    
