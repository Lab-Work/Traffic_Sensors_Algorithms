__author__ = 'Yanning Li'
"""
This file is to test wavelet approach
"""

import numpy as np
import matplotlib.pyplot as plt

from PIR import *


file_name_str = '/home/emlynlyn/Data/converted_PIR_4th_20150604_210833_037912.txt'
# t_start =


all_pir = PIR_3_MLX90620()

all_pir.read_data_from_file(file_name_str)

all_pir.pir2.plot_time_series_of_pixel(None,None,[(2,0),(2,2),(2,4),(2,6),(2,8),(2,10),(2,12),(2,14)])

# all_pir.play_video(None, None, 25, 50, 8)