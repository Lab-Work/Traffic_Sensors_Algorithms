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

file_str_list = ['./datasets/data_11_11_Noise_Analysis/PIR_8Hz_4x48_sample1.csv']

traffic_data = TrafficData_4x48()

traffic_data.read_data_file(file_str_list)

print 'Done reading data\n\n\n'

traffic_data.plot_time_series_for_pixel(None, None, [[1,8], [1, 24], [1,40]], 'raw')

traffic_data.calculate_std()


