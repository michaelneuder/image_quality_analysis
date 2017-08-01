#!/usr/bin/env python3
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image as im

def get_2d_list_slice(matrix, start_row, end_row, start_col, end_col):
    return np.asarray([row[start_col:end_col] for row in matrix[start_row:end_row]])

def get_SSIM_window(matrix, row_center, col_center, padding):
    return get_2d_list_slice(matrix, row_center-padding, row_center+padding+1, col_center-padding, col_center+padding+1)

def calculate_ssim(window_orig, window_recon):
    if window_orig.shape != (11,11) or window_recon.shape != (11,11):
        raise ValueError('please check window size for SSIM calculation!')
    else:
        orig_data = window_orig.flatten()
        recon_data = window_recon.flatten()
        k_1 = 0.01
        k_2 = 0.03
        L = 255
        mean_x = np.mean(orig_data)
        mean_y = np.mean(recon_data)
        var_x = np.var(orig_data)
        var_y = np.var(recon_data)
        covar = np.cov(orig_data, recon_data)[0][1]
        c_1 = (L*k_1)**2
        c_2 = (L*k_2)**2
        num = (2*mean_x*mean_y+c_1)*(2*covar+c_2)
        den = (mean_x**2+mean_y**2+c_1)*(var_x**2+var_y**2+c_2)
        return num/den

def main():
    print('importing image files ...')
    orig_images = np.loadtxt('../../data/sample_data/orig_3.txt')
    recon_images = np.loadtxt('../../data/sample_data/recon_3.txt')

    num_images = orig_images.shape[0]
    image_dimension = int(np.sqrt(orig_images.shape[1]))

    # reshape to add padding --- care must be taken as padding can mess things up
    orig_images = np.reshape(orig_images, [num_images,image_dimension,image_dimension])
    recon_images = np.reshape(recon_images, [num_images,image_dimension,image_dimension])

    # adding padding for SSIM calcs
    print('padding images ...')
    padding = 5
    orig_padded = []
    recon_padded = []
    for i in range(num_images):
        orig_padded.append(np.pad(orig_images[i], pad_width=padding, mode='edge'))
        recon_padded.append(np.pad(recon_images[i], pad_width=padding, mode='edge'))
    orig_padded = np.asarray(orig_padded)
    recon_padded = np.asarray(recon_padded)

    # iterating through each pixel of original image, and get 11x11 window for calculation
    SSIM_scores = np.zeros(shape=(image_dimension,image_dimension))
    for image in range(num_images):
        for row in range(padding,orig_padded.shape[1]-padding):
            for col in range(padding,orig_padded.shape[1]-padding):
                current_window_orig = get_SSIM_window(orig_padded[image], row, col, padding)
                current_window_recon = get_SSIM_window(recon_padded[image], row, col, padding)
                score = calculate_ssim(current_window_orig, current_window_recon)
                SSIM_scores[row-padding, col-padding] = score
    print(SSIM_scores.shape, SSIM_scores.mean(), SSIM_scores.std())
    hist = SSIM_scores.flatten()
    print(hist)
    plt.hist(hist, bins=1000, color='green')
    plt.xlim(0,0.05)
    plt.show()

    exit()



    upper_right_corner_orig = []
    upper_right_corner_recon = []

    # getting first SSIM window
    padding = 5
    image_index = 2
    for i in range(0,6):
        for j in range(0,6):
            print(orig_images[image_index][96*i+j], recon_images[image_index][96*i+j], combined_data[image_index][i][j])
            upper_right_corner_orig.append(orig_images[image_index][96*i+j])
            upper_right_corner_recon.append(recon_images[image_index][96*i+j])
    orig = np.pad(np.reshape(np.asarray(upper_right_corner_orig), [6,6]), pad_width=padding, mode='edge')
    recon = np.pad(np.reshape(np.asarray(upper_right_corner_recon), [6,6]), pad_width=padding, mode='edge')

    # slicing extra padding from right and bottom edges
    orig = get_2d_list_slice(orig, 0, 11, 0 , 11)
    recon = get_2d_list_slice(recon, 0, 11, 0 , 11)

    # flatten because we no longer need the structure
    orig_flat = orig.flatten()
    recon_flat = recon.flatten()

    # SSIM parameters using x=orig, y=recon for clarity -- see wikipedia on SSIM for explanation
    k_1 = 0.01
    k_2 = 0.03
    L = 255
    mean_x = np.mean(orig_flat)
    mean_y = np.mean(recon_flat)
    var_x = np.var(orig_flat)
    var_y = np.var(recon_flat)
    covar = np.cov(orig_flat, recon_flat)[0][1]
    c_1 = (L*k_1)**2
    c_2 = (L*k_2)**2

    num = (2*mean_x*mean_y+c_1)*(2*covar+c_2)
    den = (mean_x**2+mean_y**2+c_1)*(var_x**2+var_y**2+c_2)
    SSIM = num/den
    # print(SSIM)
    print(covar)

if __name__ == '__main__':
    main()
