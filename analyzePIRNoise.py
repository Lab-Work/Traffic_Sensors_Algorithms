__author__ = 'Yanning'

"""
This scripts uses the TraffcDataClasses to do quick analysis of the PIR noise
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from os.path import exists
import sys
import time

from TrafficDataClasses import *

# file_str_list_8Hz = ['./datasets/data_11_12_Noise_Analysis/PIR_32Hz_4x48.csv']
file1 = ['./datasets/data_11_13_Noise_Analysis/PIR_32Hz_1_2x16.csv']
file2 = ['./datasets/data_11_13_Noise_Analysis/PIR_32Hz_2_2x16.csv']
file3 = ['./datasets/data_11_13_Noise_Analysis/PIR_32Hz_3_2x16.csv']
file4 = ['./datasets/data_11_13_Noise_Analysis/PIR_32Hz_4_2x16.csv']

# # 8 Hz
# traffic_data_8Hz = TrafficData_4x48()
# traffic_data_8Hz.read_data_file(file_str_list_8Hz)
# # traffic_data_8Hz.plot_time_series_for_pixel(None, None, [[1,8], [1, 20], [1,40]], 'raw')
# # traffic_data_8Hz.plot_histogram_for_pixel([[1,8], [1, 20], [1,40]])
#
# # get the mu, and sigma
# mu_4x48, sigma_4x48 = traffic_data_8Hz.calculate_std()
# mu_4x48 = np.array(mu_4x48)
# mu_2d = mu_4x48.reshape((48,4)).T
# sigma_4x48 = np.array(sigma_4x48)
# sigma_2d = sigma_4x48.reshape((48,4)).T
#
# print 'mu: {0}~{1}'.format(np.min(mu_4x48), np.max(mu_4x48))
# print 'sigma: {0}~{1}'.format(np.min(sigma_4x48), np.max(sigma_4x48))
#
# traffic_data_8Hz.plot_2d_colormap(mu_2d, 18, 30, 'Mean, 8Hz, 4x48 pixels')
# traffic_data_8Hz.plot_2d_colormap(sigma_2d, 1, 6, 'Std, 8Hz, 4x48 pixels')


# 32 Hz
traffic_data_32Hz = TrafficData_2x16()

traffic_data_32Hz.read_data_file(file4)
# traffic_data_32Hz.plot_time_series_for_pixel(None, None, [[1, 20], [1, 27]], 'raw')
# traffic_data_32Hz.plot_histogram_for_pixel([[1, 20], [1, 27]])

# get the mu, and sigma
mu_2x16, sigma_2x16 = traffic_data_32Hz.calculate_std()
mu_2x16 = np.array(mu_2x16)
mu_2d = mu_2x16.reshape((16,2)).T
sigma_2x16 = np.array(sigma_2x16)
sigma_2d = sigma_2x16.reshape((16,2)).T
print 'mu: {0}~{1}'.format(np.min(mu_2x16), np.max(mu_2x16))
print 'sigma: {0}~{1}'.format(np.min(sigma_2x16), np.max(sigma_2x16))


traffic_data_32Hz.plot_2d_colormap(mu_2d, 18, 30, 'Mean, 32Hz, 2x16 pixels, covered')
traffic_data_32Hz.plot_2d_colormap(sigma_2d, 0.5, 3, 'Std, 32Hz, 2x16 pixels, covered')








plt.show()


