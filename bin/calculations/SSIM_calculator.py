#!/usr/bin/env python3
import numpy as np
import scipy as sp
from PIL import Image as im

def get_2d_list_slice(matrix, start_row, end_row, start_col, end_col):
    return np.asarray([row[start_col:end_col] for row in matrix[start_row:end_row]])

def main():
    print('importing image files...')
    orig_images = np.loadtxt('../../data/sample_data/orig_3.txt')
    recon_images = np.loadtxt('../../data/sample_data/recon_3.txt')

    num_images = orig_images.shape[0]

    # reshape to add padding --- care must be taken as padding can mess things up
    orig_images = np.reshape(orig_images, [3,96,96])
    recon_images = np.reshape(recon_images, [3,96,96])

    # adding padding for SSIM calcs
    padding = 5
    orig_padded = []
    recon_padded = []
    for i in range(num_images):
        orig_padded.append(np.pad(orig_images[i], pad_width=padding, mode='edge'))
        recon_padded.append(np.pad(recon_images[i], pad_width=padding, mode='edge'))
    orig_padded = np.asarray(orig_padded)
    recon_padded = np.asarray(recon_padded)

    


    exit()

    combined_data = np.reshape(np.dstack((orig_images, recon_images)), [3,96,96,2])
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
