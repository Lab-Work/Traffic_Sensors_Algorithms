__author__ = 'Yanning Li'
"""
This file is to test wavelet approach
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

from PIR import *


file_name_str = '/home/emlynlyn/Data/converted_PIR_4th_20150604_210833_037912.txt'
# t_start =


all_pir = PIR_3_MLX90620()

all_pir.read_data_from_file(file_name_str)

# all_pir.pir2.plot_time_series_of_pixel(None,None,[(2,0),(2,2),(2,4),(2,6),(2,8),(2,10),(2,12),(2,14)])
all_pir.pir2.plot_time_series_of_pixel(None,None,[(2,0),(2,8),(2,15)])

# all_pir.play_video(1433466270, 1433466300, 25, 50, 8)
# all_pir.play_video(None, None, 15, 40, 8)

# all_pir.pir2.plot_agg_temp([(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(2,11),(2,12),(2,13),(2,14),(2,15)], None, None)


# all_pir.pir2.plot_agg_temp([(0,8),(1,8),(2,8),(3,8)], None, None)

# # Here is a very quick test of the wavelet
# # Use the PIR2, pixel (2,6)
# pir2_pixel_data = all_pir.pir2.all_temperatures[6*4+2,:]
# cA, cD = pywt.dwt(pir2_pixel_data, 'db1')
#
# fig, ax = plt.subplots(3,1)
# ax[0].plot(all_pir.pir2.time_stamps, pir2_pixel_data)
# ax[0].set_xlim([all_pir.pir2.time_stamps[0], all_pir.pir2.time_stamps[-1]])
# ax[0].set_title('discrete wavelet transform')
# ax[1].plot(cA)
# ax[1].set_xlim([0,len(cA)])
# ax[2].plot(cD)
# ax[2].set_xlim([0,len(cD)])
# plt.draw()
#
#
# # a multi-level decomposition
# cA3, cD3, cD2, cD1 = pywt.wavedec(pir2_pixel_data, 'db1', level=3)
#
# fig1, ax1 = plt.subplots(2,1)
# ax1[0].plot(all_pir.pir2.time_stamps, pir2_pixel_data)
# ax1[0].set_xlim([all_pir.pir2.time_stamps[0], all_pir.pir2.time_stamps[-1]])
# ax1[0].set_title('discrete wavelet transform')
# ax1[1].plot(cA3)
# ax1[1].set_xlim([0,len(cA3)])
# plt.draw()
#
# fig2, ax2 = plt.subplots(3,1)
# ax2[0].plot(cD3)
# ax2[0].set_xlim([0,len(cD3)])
# ax2[0].set_ylabel('cD3')
# ax2[1].plot(cD2)
# ax2[1].set_xlim([0,len(cD2)])
# ax2[1].set_ylabel('cD2')
# ax2[2].plot(cD1)
# ax2[2].set_xlim([0,len(cD1)])
# ax2[2].set_ylabel('cD1')
# plt.show()


