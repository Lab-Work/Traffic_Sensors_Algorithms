__author__ = 'Yanning'


"""
This script does a quick plot of 1x16 data.
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
file1 = ['./datasets/data_11_20/PIR_2015_11_20__11_35_03_1x16.csv']
file2 = ['./datasets/data_11_20/PIR_2015_11_20__11_40_04_1x16.csv']
file3 = ['./datasets/data_11_20/PIR_2015_11_20__11_45_05_1x16.csv']
file4 = ['./datasets/data_11_20/PIR_2015_11_20__11_50_05_1x16.csv']
file5 = ['./datasets/data_11_20/PIR_2015_11_20__11_55_06_1x16.csv']
file6 = ['./datasets/data_11_20/PIR_2015_11_20__12_00_06_1x16.csv']



pir_data = TrafficData_1x16()

pir_data.read_data_file(file_name_str=file1, datestr='2015_11_20')

pir_data.plot_heat_map_in_period(t_start_str='2015_11_20_11_37_10_0',
                                 t_end_str='2015_11_20_11_37_16_000000',
                                 T_min=15, T_max=30, data_option='raw')

pir_data.plot_time_series_for_pixel(t_start_str=None,
                                    t_end_str=None,
                                    pixel_list=[(1,7)],
                                    data_option='raw')

pir_data.subtract_background(80*5)


pir_data.plot_time_series_for_pixel(t_start_str=None,
                                    t_end_str=None,
                                    pixel_list=[(1,7)],
                                    data_option='cleaned')

pir_data.plot_heat_map_in_period(t_start_str='2015_11_20_11_37_10_0',
                                 t_end_str='2015_11_20_11_37_16_000000',
                                 T_min=-6, T_max=9, data_option='cleaned')



plt.show()


