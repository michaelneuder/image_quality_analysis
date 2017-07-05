#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (13,10)
plt.rcParams['font.size'] = 18
import csv
from scipy import stats

def main():
    prediction_plot = []
    target_plot = []
    with open('../nets/wip/results/post_training_prediction.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            prediction_plot.append(float(line[0]))
    csv_file.close()
    prediction_plot = np.asarray(prediction_plot)
    with open('../nets/wip/results/post_training_target.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            target_plot.append(float(line[0]))
    csv_file.close()
    target_plot = np.asarray(target_plot)

    slope, intercept, r_value, p_value_, std_err = stats.linregress(prediction_plot, target_plot)

    plt.plot(prediction_plot, target_plot, 'r.', alpha=.25)
    plt.title('prediction vs target')
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.text(0.7, -0.3, 'r-squared = {0:.5f}'.format(r_value**2), style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    # plt.show()

    mean = target_plot.mean()
    differences_squared = []
    differences = []
    for i in range(len(target_plot)):
        difference = target_plot[i] - mean
        differences.append(abs(difference))
        differences_squared.append(difference**2)
    differences = np.asarray(differences)
    differences_squared = np.asarray(differences_squared)
    print(differences.mean(), differences_squared.mean())


if __name__ == '__main__':
    main()
