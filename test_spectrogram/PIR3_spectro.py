'''
Author: Yanning Li
Date: 05/28/2015

input: four columns: time, pir1, pir2, pir3
    time is a float number in unix or matlab
    pir123 are temperatures in C

This class specifically deals with the 3-PIR sensor,
plot the spectrogram, manipulate data, try different possible outlier detection solutions.
'''

import csv
import numpy as np
# from sklearn import linear_model
# from sklearn import cross_validation
import matplotlib.pyplot as plt
from datetime import datetime
# from astroML.plotting import setup_text_plots
# setup_text_plots(fontsize=14, usetex=True)

# define constant
MATLAB_2_SEC = 86400


class PIR3_spectro:
    
    def __init__(self, filename):

        file = open(filename, 'r')
        dataset = csv.reader(file)

        self.time = []
        self.pir1 = []
        self.pir2 = []
        self.pir3 = []

        # read file
        for line in dataset:
            self.time.append(float(line[0]))
            self.pir1.append(float(line[1]))
            self.pir2.append(float(line[2]))
            self.pir3.append(float(line[3]))

        file.close()

        # convert to readable relative time in s
        self.time = np.array(self.time)
        self.time = (self.time - self.time[0])*MATLAB_2_SEC


    # a function for performing the spectrogram
    def plot_spectro(self, pir_id, window_size, step_size):
        pass

    # plot the time series data
    # pir_id: 1,2,3 or all
    def plot_data(self, pir_id):
        plt.figure()

        if pir_id == 'all':
            pir_id = [1,2,3]

        for i in pir_id:
            if i == 1:
                plt.plot(self.time, self.pir1)
            elif i == 2:
                plt.plot(self.time, self.pir2)
            elif i == 3:
                plt.plot(self.time, self.pir3)

        plt.title('Time Series of PIR Signal')
        plt.ylabel('Temperature ($^{\circ}C$)')
        plt.xlabel('Time (sec)')
        plt.show()

