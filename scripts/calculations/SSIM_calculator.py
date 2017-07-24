#!/usr/bin/env python3
import numpy as np
import scipy as sp
import PIL.Image as im

def main():
    print('importing image files...')
    original_images = np.loadtxt('../../data/sample_data/orig_3.txt')
    reconstructed_images = np.loadtxt('../../data/sample_data/recon_3.txt')
    print(original_images.shape)
    original_images = original_images.reshape(3,96,96)
    reconstructed_images = reconstructed_images.reshape(3,96,96)

    image1 = np.asarray(original_images[1], dtype='uint8')
    image2 = np.asarray(reconstructed_images[1], dtype='uint8')
    image_view = im.fromarray(image1, 'L')
    image_view.show()
    image_view = im.fromarray(image2, 'L')
    image_view.show()



if __name__ == '__main__':
    main()
