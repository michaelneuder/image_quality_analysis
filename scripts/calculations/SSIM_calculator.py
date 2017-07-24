#!/usr/bin/env python3
import numpy as np
import scipy as sp

def main():
    print('importing image files...')
    original_images = np.loadtxt('../../data/sample_data/orig_3.txt')
    reconstructed_images = np.loadtxt('../../data/sample_data/recon_3.txt')
    combined_data = np.reshape(np.dstack((original_images, reconstructed_images)), [3,96,96,2])
    upper_right_corner_orig = []
    upper_right_corner_recon = []
    for i in range(0,6):
        for j in range(0,6):
            print(original_images[2][96*i+j], reconstructed_images[2][96*i+j], combined_data[2][i][j])
            upper_right_corner_orig.append(original_images[2][96*i+j])
            upper_right_corner_recon.append(reconstructed_images[2][96*i+j])

if __name__ == '__main__':
    main()
