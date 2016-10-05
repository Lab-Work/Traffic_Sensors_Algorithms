from TrafficDataClass import *
import time
from datetime import timedelta
import glob

"""
This script is used to visualize the PIR data collected by MLX90621 60 and 120 FOV. Explore the speed estimation
algorithm as well.
"""
__author__ = 'Yanning Li'

# ================================================================================
# configuration
# read the data
folder = '0915_2016'
# folder = '0927_2016'
# folder = '0928_2016'
# folder = '1004_2016'

dataset = '0915_128hz_corrected'
# dataset = '0927_ff_128hz'
# dataset = '0928_ff_128hz'
# dataset = '64_600_1'

fps = 128

data_key = 'temp_data'

# ================================================================================
temp_data_file = '../datasets/{0}/{1}.npy'.format(folder, dataset)
#
data = TrafficData()
# dt = timedelta(seconds=3)

periods = data.get_data_periods('../datasets/{0}/*.npy'.format(folder))

data.load_npy_data(file_name_str=temp_data_file, dataset=dataset, data_key='temp_data')
# data.plot_sample_timing(dataset=dataset)


# data.down_sample(dataset=dataset, data_key=data_key,
#                  from_freq=128, to_freq=8, save_name='../datasets/data_09_27_2016/927ff1temp8hz.npy')



# periods = data.get_data_periods('../datasets/1004_2016/*.npy'.format(folder))
#
# all_files = glob.glob('../datasets/1004_2016/*.npy')
#
# for f in all_files:
#
#     dataset = f.strip().split('/')[-1].replace('.npy','')
#
#     data.load_npy_data(file_name_str=f, dataset=dataset, data_key='temp_data')
#
#     data.check_duplicates(dataset=dataset, data_key='temp_data',
#                             t_start=periods[dataset][0], t_end=periods[dataset][1])

# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.3,0.1),
#                       pixel=(2,11), window_s=10, step_s=1.0/16.0, fps=fps)

# mu, sigma, noise_mu, noise_sigma = data.get_noise_distribution(dataset=dataset, data_key='temp_data',
#                             t_start=periods[dataset][0]+timedelta(seconds=3),
#                             t_end=periods[dataset][0]+timedelta(seconds=180),
#                             p_outlier=0.01, stop_thres=(0.1, 0.01),pixels=None, suppress_output=True)
#
# data.cus_imshow(data_to_plot=mu, cbar_limit=None, title='mean for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_mu'.format(dataset))
# data.cus_imshow(data_to_plot=sigma, cbar_limit=None, title='std for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_std'.format(dataset))
# data.cus_imshow(data_to_plot=noise_mu,cbar_limit=None,title='noise mean for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_noise_mu'.format(dataset))
# data.cus_imshow(data_to_plot=noise_sigma,cbar_limit=None,title='noise std for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_noise_std'.format(dataset))
#


# data.plot_histogram_for_pixel(dataset=dataset, data_key='temp_data',pixels=[(2,11)],
#                               t_start=periods[dataset][0]+timedelta(seconds=3),
#                               t_end=periods[dataset][0]+timedelta(seconds=153))




data.normalize_data(dataset=dataset, data_key=data_key, norm_data_key=None,
                    t_start=None, t_end=None)
#
# data.plot_heat_map_in_period(dataset=dataset, data_key='norm_'+data_key,
#                              t_start= periods[dataset][0]+timedelta(seconds=35),
#                              t_end=periods[dataset][0]+timedelta(seconds=105),
#                              cbar_limit=(2,6), option='vec', nan_thres=2.038)
#

vehicles = [(36.5, 38), (38, 39.2), (40, 42), (42, 44), (43.5, 45), (59, 61), (86, 88), (98.5, 100)]

for veh in vehicles:
    data.get_img_in_period(dataset=dataset, data_key='norm_'+data_key,
                           t_start=periods[dataset][0]+timedelta(seconds=veh[0]),
                           t_end = periods[dataset][0]+timedelta(seconds=veh[1]),
                           cbar_limit=(0,1),option='vec', nan_thres=2.71, plot=False,
                           folder='../figs/hough_realdata/')

# data.get_veh_img_in_period(dataset=dataset, data_key='norm_'+data_key,
#                                 t_start=periods[dataset][0]+timedelta(seconds=35),
#                                 t_end=periods[dataset][0]+timedelta(seconds=105),
#                                 cbar_limit=[0, 5], option='vec',
#                                 nan_thres=None, plot=False,
#                                   dur=1, folder='../figs/hough_realdata/', fps=128)

# img[~np.isnan(img)] = 1
# img[np.isnan(img)] = 0
#
# np.save('img1.npy', img)


# data.plot_heat_map_in_period(dataset=dataset, data_key='norm_'+ data_key,
#                              t_start= periods[dataset][0]+dt, t_end=periods[dataset][1],
#                              cbar_limit=(2.0, 6), option='tworow', nan_thres=2.038)
#
# data.plot_heat_map_in_period(dataset=dataset, data_key='norm_'+ data_key,
#                              t_start= periods[dataset][0]+dt, t_end=periods[dataset][1],
#                              cbar_limit=(2.0, 6), option='tworow', nan_thres=2.71)
plt.show()

# data.save_as_avi(sensor_id=sensor_id, data_type='norm_temp_data', fps=16)

# pixel = (2,11)
# data.plot_time_series_for_pixel(dataset=dataset, data_key=data_key, t_start=None, t_end=None, pixels=[pixel])






