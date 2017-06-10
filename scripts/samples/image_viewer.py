#!/usr/bin/env python3
import numpy as np
from PIL import Image as im

def main():
    '''
    just to look at a couple of the images.
    '''
    original_images = np.loadtxt("../../data/sample_data/orig_3pics.txt")
    reconstructed_images = np.loadtxt("../../data/sample_data/recon_3pics.txt")
    original_images = original_images.reshape(3,96,96)
    reconstructed_images = reconstructed_images.reshape(3,96,96)

    image = np.asarray(original_images[0], dtype='uint8')
    # image = np.asarray(reconstructed_images[1], dtype='uint8')
    image_view = im.fromarray(image, 'L')
    image_view.show()

if __name__ == '__main__':
    main()
