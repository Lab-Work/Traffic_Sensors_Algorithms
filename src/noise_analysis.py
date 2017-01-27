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
# ================================================================================
# read the data
# folder = '0915_2016'
# folder = '0927_2016'
# folder = '0928_2016'
# folder = '1004_2016'
# folder = '../datasets/1005_2016/'
# folder = '../datasets/1006_2016/'
folder = '../datasets/1013_2016/'

# dataset = '0915_128hz_corrected'
# dataset = '0927_ff_128hz'
# dataset = '0928_ff_128hz'
# dataset = '64_600_0_s2'
# dataset = '64_300_0_150142'
# dataset = '64_300_0_152445'
# dataset = '64_180_0_161152'
# dataset = '128_300_0_150821p1'
# dataset = '128_300_0_150821p2'

# dataset = '64_300_0_150142'
# dataset = '64_180_0_c2_s1'
# dataset = '64_180_1_c2_s2'
# dataset = '64_180_0_c2_s3'
# dataset = '64_180_0_210838'
# dataset = '64_180_20_c2_s2'
# dataset = '64_180_0_s2_211656'
# dataset = '64_600_1_c2_s1'
# dataset = '64_180_10_c2_s1'
# dataset = '64_180_0_184109_c1s1_c2s2'
# dataset = '64_300_0_163435'


# ======================================================
# dataset = '1310-205905' # initial 10 second
# dataset = '1310-213154'
# dataset='1310-210300'
dataset='1310-221543'

fps = 64

data_key = 'temp_data'

# ================================================================================
temp_data_file = folder + '{0}.txt'.format(dataset)

data = TrafficData()
# dt = timedelta(seconds=3)

periods = data.get_txt_data_periods(folder+'*.txt', update=True)

data.load_txt_data(file_name_str=temp_data_file, dataset=dataset, data_key='temp_data')


# ================================================================================
# Check sampling speed
# ================================================================================
# data.plot_sample_timing(dataset=dataset)


# ================================================================================
# Plot the evolution of noise for a pixel
# ================================================================================

# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(0,0), window_s=10, step_s=1.0/16.0, fps=fps)
#
# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(0,15), window_s=10, step_s=1.0/16.0, fps=fps)

data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
                      pixel=(1,5), window_s=5, step_s=1.0/16.0, fps=fps)

data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
                      pixel=(1,20), window_s=5, step_s=1.0/16.0, fps=fps)
#
# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(2,10), window_s=5, step_s=1.0/16.0, fps=fps)

# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(3,0), window_s=10, step_s=1.0/16.0, fps=fps)
#
# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(3,15), window_s=10, step_s=1.0/16.0, fps=fps)



# ================================================================================
# Check the noise mean and standard deviation
# They are saved in ../figs/
# ================================================================================
# mu, sigma, noise_mu, noise_sigma = data.get_noise_distribution(dataset=dataset, data_key='temp_data',
#                             t_start=periods[dataset][0]+timedelta(seconds=3),
#                             t_end=periods[dataset][1],
#                             p_outlier=0.01, stop_thres=(0.1, 0.01),pixels=None, suppress_output=True)
# # Overall mean and std
# data.cus_imshow(data_to_plot=mu, cbar_limit=None, title='mean for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_mu'.format(dataset))
# data.cus_imshow(data_to_plot=sigma, cbar_limit=None, title='std for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_std'.format(dataset))
#
# noise mean and std
# data.cus_imshow(data_to_plot=noise_mu,cbar_limit=None,title='noise mean for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_noise_mu'.format(dataset))
# data.cus_imshow(data_to_plot=noise_sigma,cbar_limit=None,title='noise std for {0}'.format(dataset),
#                 annotate=True, save_name='{0}_noise_std'.format(dataset))



# ================================================================================
# Plot the histogram of a pixel during a peirod.
# ================================================================================
# data.plot_histogram_for_pixel(dataset=dataset, data_key='temp_data',pixels=[(1,8)],
#                               t_start=periods[dataset][0]+timedelta(seconds=33),
#                               t_end=periods[dataset][0]+timedelta(seconds=633))


# ================================================================================
# Normalize the data and visualize in heat maps
# ================================================================================
# data.normalize_data(dataset=dataset, data_key=data_key, norm_data_key=None,
#                     t_start=None, t_end=None)

# data.plot_heat_map_in_period(dataset=dataset, data_key='norm_'+data_key,
#                              t_start= periods[dataset][0]+timedelta(seconds=33),
#                              t_end=periods[dataset][0]+timedelta(seconds=48),
#                              cbar_limit=(2,6), option='vec', nan_thres=2.038)



# img = data.get_heat_img_in_period(dataset=dataset, data_key='norm_'+data_key,
#                                 t_start=periods[dataset][0]+timedelta(seconds=35),
#                                 t_end=periods[dataset][0]+timedelta(seconds=65),
#                                 cbar_limit=[0, 5], option='vec',
#                                 nan_thres=None, plot=False,
#                                   dur=1, folder='../figs/hough_realdata/')


# ================================================================================
# Plot the time series data for a pixel
# ================================================================================
# data.plot_time_series_for_pixel(dataset=dataset, data_key=data_key, t_start=None, t_end=None, pixels=[(1,8)])



# ================================================================================
# Downsample data
# ================================================================================
# data.down_sample(dataset=dataset, data_key=data_key,
#                  from_freq=128, to_freq=8, save_name='../datasets/data_09_27_2016/927ff1temp8hz.npy')


# ================================================================================
# Check duplicates
# ================================================================================
# all_files = glob.glob('../datasets/1004_2016/*.npy')
# for f in all_files:
#
#     dataset = f.strip().split('/')[-1].replace('.npy','')
#
#     data.load_npy_data(file_name_str=f, dataset=dataset, data_key='temp_data')
#
#     data.check_duplicates(dataset=dataset, data_key='temp_data',
#                             t_start=periods[dataset][0], t_end=periods[dataset][1])


plt.show()