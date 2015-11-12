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

file_str_list_8Hz = ['./datasets/data_11_11_Noise_Analysis/PIR_8Hz_4x48_sample1.csv']
file_str_list_32Hz = ['./datasets/data_11_11_Noise_Analysis/PIR_32Hz_2x16_sample2.csv']

# 8 Hz
traffic_data_8Hz = TrafficData_4x48()
traffic_data_8Hz.read_data_file(file_str_list_8Hz)
traffic_data_8Hz.plot_time_series_for_pixel(None, None, [[1,8], [1, 24], [1,40]], 'raw')
traffic_data_8Hz.plot_histogram_for_pixel([[1,8], [1, 24], [1,40]])


# 32 Hz
traffic_data_32Hz = TrafficData_2x16()
traffic_data_32Hz.read_data_file(file_str_list_32Hz)
traffic_data_32Hz.plot_time_series_for_pixel(None, None, [[1, 20], [1, 27]], 'raw')
traffic_data_32Hz.plot_histogram_for_pixel([[1, 20], [1, 27]])

plt.show()


