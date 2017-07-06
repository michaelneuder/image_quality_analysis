#!/usr/bin/env python3
import numpy as np

def main():
    print('generating random images ... ')
    rand_img_1 = np.random.random_sample((96,96))
    rand_img_2 = np.random.random_sample((96,96))
    difference = abs(rand_img_1 - rand_img_2)
    print(rand_img_1[0][1], rand_img_2[0][1], difference[0][1])
    print(difference.shape)


if __name__ == '__main__':
    main()
