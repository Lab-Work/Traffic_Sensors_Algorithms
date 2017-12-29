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
# dataset = '60fov128hztemp'
# dataset = '60fov128hzp2'
# dataset = '60fov128hzp2'
# dataset = 'sdtest1'
dataset = '927ff1temp'
# dataset ='sweep_data'

data_key = 'temp_data'

# temp_data_file = '../datasets/data_09_15_2016/{0}.npy'.format(dataset)
temp_data_file = '../datasets/data_09_27_2016/{0}.npy'.format(dataset)
# temp_data_file = '../datasets/data_09_28_2016_office/sdtest1.npy'
# temp_data_file = '../datasets/data_10_02_2016/sweep_data.npy'



# ================================================================================
data = TrafficData()
# dt = timedelta(seconds=3)

# periods = data.get_data_periods('../datasets/data_09_15_2016/*.npy')
# periods = data.get_data_periods('../datasets/data_09_27_2016/*.npy')
# periods = data.get_data_periods('../datasets/data_09_28_2016_office/*.npy')
# periods = data.get_data_periods('../datasets/data_10_02_2016/sweep_data.npy')
# periods['sdtest1'] = (periods['sdtest1'][0] + timedelta(seconds=10),
#                          periods['sdtest1'][1])

# data.load_npy_data(file_name_str=temp_data_file, dataset=dataset, data_key='temp_data')

# print (periods[dataset][1]-periods[dataset][0])
# print data.PIR[dataset]['temp_data'].shape


# data.save_as_avi(sensor_id=sensor_id, data_type='norm_temp_data', fps=16)

video_time_str = '2016-09-27 16:39:21.057665'
video_time = data.video_string_to_time(video_time_str)
dt1 = timedelta(seconds=380)
dt2 = timedelta(seconds=400)
#
data.trim_video(input_video='../datasets/data_09_27_2016/2016-09-27 16_39_21.057665.mp4',
                input_timestamp=video_time,
                trim_period=(video_time+dt1, video_time+dt2),
                output_video='../datasets/data_09_27_2016/sep27_20s.avi')

# data.play_video(input_video='../datasets/data_09_27_2016/2016-09-27 16_39_21.057665.mp4',
#                 input_timestamp=video_time,
#                 trim_period=(video_time+dt1, video_time+dt2))






