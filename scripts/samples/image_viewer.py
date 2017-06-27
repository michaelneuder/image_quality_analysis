#!/usr/bin/env python3
import numpy as np
from PIL import Image as im

def main():
    '''
    just to look at a couple of the images.
    '''
    original_images = np.loadtxt("../../data/sample_data/orig_15.txt")
    reconstructed_images = np.loadtxt("../../data/sample_data/recon_15.txt")
    original_images = original_images.reshape(15,96,96)
    reconstructed_images = reconstructed_images.reshape(15,96,96)

    image1 = np.asarray(original_images[14], dtype='uint8')
    image2 = np.asarray(reconstructed_images[14], dtype='uint8')
    image_view = im.fromarray(image1, 'L')
    image_view.show()
    image_view = im.fromarray(image2, 'L')
    image_view.show()


if __name__ == '__main__':
    main()
