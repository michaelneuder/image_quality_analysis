#!/usr/bin/env python3
import imageio


def main():
    images = []
    for ii in range(300):
        images.append(imageio.imread('single_epoch/prediction_demo{}.png'.format(ii)))
    imageio.mimsave('movie.gif', images)



if __name__ == '__main__':
    main()
