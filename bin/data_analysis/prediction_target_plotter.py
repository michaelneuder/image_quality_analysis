#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (13,10)
plt.rcParams['font.size'] = 18
import csv
import scipy.stats as stats

def main():
    prediction_plot = []
    target_plot = []
    with open('../nets/wip/post_training.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        line_count = 0
        for line in reader:
            if line_count > 0:
                prediction_plot.append(float(line[1]))
                target_plot.append(float(line[0]))
            line_count+=1
    csv_file.close()
    prediction_plot = np.asarray(prediction_plot)
    target_plot = np.asarray(target_plot)

    slope, intercept, r_value, p_value_, std_err = stats.linregress(prediction_plot, target_plot)

    plt.plot(prediction_plot, target_plot, 'r.', alpha=.25)
    plt.title('prediction vs target')
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.text(0.12, .9, 'r-squared = {0:.5f}'.format(r_value**2), style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    plt.show()

    mean = target_plot.mean()
    differences_squared = []
    differences = []
    for i in range(len(target_plot)):
        difference = target_plot[i] - mean
        differences.append(abs(difference))
        differences_squared.append(difference**2)
    differences = np.asarray(differences)
    differences_squared = np.asarray(differences_squared)
   
    x_plot = np.arange(len(target_plot))
    plt.plot(x_plot, prediction_plot, 'r+')
    plt.xlabel('pixel')
    plt.ylabel('prediction')
    plt.show()

    n, bins, patches = plt.hist(prediction_plot, 50, normed=1, facecolor='blue', alpha=0.5, label='prediction')
    n, bins, patches = plt.hist(target_plot, 50, normed=1, facecolor='red', alpha=0.5, label='target')
    plt.xlabel('difference')
    plt.ylabel('quantity')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
