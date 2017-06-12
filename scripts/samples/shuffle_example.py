#!/usr/bin/env python3
import numpy as np
from PIL import Image as im

def main():
    original_images = np.loadtxt('../../data/sample_data/orig_500.txt')
    reconstructed_images = np.loadtxt('../../data/sample_data/recon_500.txt')
    comparison_images = np.loadtxt('../../data/sample_data/comp_500.txt')

    batch_indices = np.arange(500)
    np.random.shuffle(batch_indices)
    original_batch = []
    reconstructed_batch = []
    comparison_batch = []
    for i in batch_indices[:50]:
        original_batch.append(original_images[i])
        reconstructed_batch.append(reconstructed_images[i])
        comparison_batch.append(comparison_images[i])

    # verify images
    original_batch = np.asarray(original_batch)
    reconstructed_batch = np.asarray(reconstructed_batch)
    comparison_batch = np.asarray(comparison_batch)

    original_batch = original_batch.reshape(original_batch.shape[0],96,96)
    reconstructed_batch = reconstructed_batch.reshape(reconstructed_batch.shape[0],96,96)

    image = np.asarray(original_batch[49], dtype='uint8')
    image1 = np.asarray(reconstructed_batch[49], dtype='uint8')
    image_view = im.fromarray(image, 'L')
    image_view1 = im.fromarray(image1, 'L')
    image_view.show()
    image_view1.show()


if __name__ == '__main__':
    main()
