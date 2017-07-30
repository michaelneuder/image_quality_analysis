#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    rand_array1 = np.random.random_sample(1000)
    rand_array2 = np.random.random_sample(1000)
    diff = np.asarray(rand_array1 - rand_array2)
    for i in range(len(diff)):
        diff[i] = diff[i]**2
    x_plot = np.arange(1000)
    plt.plot(x_plot, diff)
    plt.show()
    print('random samples are on average {} far apart'.format(diff.mean()))

if __name__ == '__main__':
    main()
