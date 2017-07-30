#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14,6)
plt.rcParams['font.size'] = 18
import csv

def main():
    x_plot = []
    y_plot = []
    with open('../../data/results/one_sample.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        line_count = 0
        for line in reader:
            if line_count > 0:
                x_plot.append(line[0])
                y_plot.append(line[1])
            line_count+=1
    csv_file.close()
    plt.ylim(0,1)
    plt.plot(x_plot, y_plot, 'r.--', alpha=.25)
    plt.title('error over training')
    plt.xlabel('training step')
    plt.ylabel('error [avg_pixel_diff^2]')
    plt.show()

if __name__ == '__main__':
    main()
