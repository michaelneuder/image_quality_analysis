#!/usr/bin/env python3

# -----------
# convolution
# -----------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from scipy import signal as sig

def main():
    # convolution inverts the second argument and slides it along the first
    print("\nconvolution")
    x = [3, 4, 5]
    h = [2, 1, 0]
    y = np.convolve(x,h)
    print(y, "\n")

    # we can convolve with more dimension
    print("2d convolution (w/ zero padding)")
    mat = [[1,2,3],[4,5,6],[7,8,9]]
    ker = [[-1,1]]
    y = sig.convolve(mat,ker)
    print(y, '\n')

    # valid flag allows for only items that dont rely on padding
    print("2d convolution (w/o zero padding)")
    mat = [[1,2,3],[4,5,6],[7,8,9]]
    ker = [[-1,1]]
    y = sig.convolve(mat,ker,'valid')
    print(y, '\n')

    print("2d convolution (w/ zero padding) on a 2d kernel")
    ker_2d = [[-1,1],[2,-2]]
    y = sig.convolve(mat, ker_2d)
    print(y, '\n')

    print("2d convolution (w/0 zero padding) on a 2d kernel")
    ker_2d = [[-1,1],[2,-2]]
    y = sig.convolve(mat, ker_2d, 'valid')
    print(y, '\n')


if(__name__ == '__main__'):
    main()
