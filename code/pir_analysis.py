from TrafficDataClass import *
import time
from datetime import timedelta

"""
This script is used to visualize the PIR data collected by MLX90621 60 and 120 FOV. Explore the speed estimation
algorithm as well.
"""
__author__ = 'Yanning Li'

# ================================================================================
# configuration
# read the data
sensor_id = '60fov32hz'

raw_data_file = '../datasets/data_09_15_2016/{0}.npy'.format(sensor_id)
temp_data_file = '../datasets/data_09_15_2016/{0}temp.npy'.format(sensor_id)



# ================================================================================

data = TrafficData()
dt = timedelta(seconds=1)

periods = data.get_data_periods('../datasets/data_09_15_2016/*.npy')

data.normalize_data([temp_data_file], periods=periods, skip_dt=1)



# data.load_npy_data(file_name_str=raw_data_file, sensor_id=sensor_id, data_type='raw_data')
# data.load_npy_data(file_name_str=temp_data_file, sensor_id=sensor_id, data_type='temp_data')

# data.plot_time_series_for_pixel(t_start_str=data.time_to_string(periods[sensor_id][0] + dt),
#                               t_end_str=data.time_to_string(periods[sensor_id][1]),
#                               pixel_list=[(sensor_id, [(0,1),(2,6)])],data_type='temp_data')
# data.plot_time_series_for_pixel(t_start_str=data.time_to_string(periods[sensor_id][0] + dt),
#                               t_end_str=data.time_to_string(periods[sensor_id][1]),
#                               pixel_list=[(sensor_id, [(0,1),(2,6)])],data_type='raw_data')
#
# data.plot_histogram_for_pixel(t_start_str=data.time_to_string(periods[sensor_id][0] + dt),
                              # t_end_str=data.time_to_string(periods[sensor_id][1]),
                              # pixel_list=[(sensor_id, [(0,1),(2,6)])],data_type='raw_data')

# data.plot_histogram_for_pixel(t_start_str=data.time_to_string(periods[sensor_id][0] + dt),
#                               t_end_str=data.time_to_string(periods[sensor_id][1]),
#                               pixel_list=[(sensor_id, [(0,1),(2,6)])],data_type='temp_data')




# data.plot_time_series_for_pixel(t_start_str=t_s, t_end_str=t_e, pixel_list=[(sensor_id,[(0,4),(1,8)])],data_type='temp_data')

# data.plot_histogram_for_pixel(t_start_str=t_s, t_end_str=t_e,
#                               pixel_list=[(sensor_id, [(0,4), (1,8)])], data_type='temp_data')

# data.subtract_PIR_background(pir_mxn=sensor_id, background_duration=100,
#                                      save_in_file_name=None, data_type='temp_data')

# data.plot_heat_map_in_period(sensor_id=sensor_id, data_type='temp_data',
#                                      t_start_str=t_s, t_end_str=t_e,
#                                      T_min=15, T_max=35)


plt.show()

