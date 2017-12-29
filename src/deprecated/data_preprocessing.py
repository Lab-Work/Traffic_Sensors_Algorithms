from TrafficDataClass import *
import time
from datetime import timedelta
import glob

"""
This script is used to visualize the datasets collected in the field: 1013, 1027, 1103, 1116, 1118
"""
__author__ = 'Yanning Li'


# ================================================================================
# configuration
# ================================================================================

# --------------------------------------------------
# dataset 1003:
# - Neil street. Freeflow and stop-and-to in total 33 min
# - One PIR sensor array 4x32, at 64 Hz
# --------------------------------------------------
folder = '../datasets/1013_2016/'
# dataset = '1310-205905'   # initial 10 second
dataset = '1310-213154'   # freeflow part 1
# dataset='1310-210300'     # freeflow part 2
# dataset='1310-221543'       # stop and go
fps = 64
data_key = 'raw_data'

# --------------------------------------------------
# dataset 1027:
# - Neil street. Two PIR sensors
# - One PIR sensor array 4x32, at 64 Hz
# --------------------------------------------------
# folder = '../datasets/1027_2016/senspi1/'
# dataset = '2710-192614'   # 7 seconds
# dataset = '2710-205846'     # 45 min

# folder = '../datasets/1027_2016/senspi2/'
# dataset = '2710-210055'   # 45 min

# fps = 64
# data_key = 'raw_data'


# --------------------------------------------------
# dataset 1103:
# - Neil street. Three PIR sensor arraies
# - One PIR sensor array 4x32, at 64 Hz
# --------------------------------------------------
# folder = '../datasets/1103_2016/s1/'
# dataset = '0311-191717'

# folder = '../datasets/1103_2016/s2/'
# dataset = '0311-192414'

# folder = '../datasets/1103_2016/s3/'
# dataset = '0311-193053'
#
# fps = 64
# data_key = 'raw_data'


# --------------------------------------------------
# dataset 1116:
# - Neil street. Three PIR sensor arraies
# - One PIR sensor array 4x32, at 64 Hz
# --------------------------------------------------
# folder = '../datasets/1116_2016/s1/'
# dataset = '1611-061706'

# folder = '../datasets/1116_2016/s2/'
# dataset = '0711-211706'

# folder = '../datasets/1116_2016/s3/'
# dataset = '1611-151742'
#
# fps = 64
# data_key = 'raw_data'


# --------------------------------------------------
# dataset 1118:
# - Neil street. Three PIR sensor arrays, for stop and go data.
# - One PIR sensor array 4x32, at 64 Hz
# --------------------------------------------------
# folder = '../datasets/1118_2016/s1/'
# dataset = '1611-171706'

# folder = '../datasets/1118_2016/s2/'
# dataset = '1811-144926'

# folder = '../datasets/1118_2016/s3/'
# dataset = '1611-171707'

# fps = 64
# data_key = 'raw_data'
#

# ================================================================================
temp_data_file = folder + '{0}.txt'.format(dataset)
#
data = TrafficData()
#
periods = data.get_txt_data_periods(folder+'*.txt', update=True)
#
data.load_txt_data(file_name_str=temp_data_file, dataset=dataset, data_key=data_key)


# ================================================================================
# Check sampling frequency
# ================================================================================
# data.plot_sample_timing(data_key=data_key)


# ================================================================================
# Plot the evolution of noise for a pixel
# ================================================================================

# data.plot_noise_evolution(data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(0,8), window_s=60, step_s=10, fps=fps)
# data.plot_noise_evolution(data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(1,8), window_s=60, step_s=10, fps=fps)
# data.plot_noise_evolution(data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(2,8), window_s=60, step_s=10, fps=fps)
# data.plot_noise_evolution(data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(3,8), window_s=60, step_s=10, fps=fps)
#
# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(0,15), window_s=10, step_s=1.0/16.0, fps=fps)

# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(1,5), window_s=5, step_s=1.0/16.0, fps=fps)
#
# data.plot_noise_evolution(dataset=dataset, data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(1,20), window_s=5.0, step_s=1.0, fps=fps)
#
# data.plot_noise_evolution(data_key=data_key, p_outlier=0.01, stop_thres=(0.1,0.01),
#                       pixel=(2,10), window_s=5, step_s=1.0, fps=fps)

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
data.plot_heat_map_in_period(data_key=data_key, t_start=periods[dataset][0], t_end=periods[dataset][1],
                             cbar_limit=(20,40), option='vec', nan_thres_p=None, plot=True, folder='../workspace/',
                             save_img=False, save_npy=False)
# data.plot_heat_map_in_period(data_key=data_key, t_start=periods[dataset][0], t_end=periods[dataset][1],
#                              cbar_limit=(20,40), option='3rd', nan_thres_p=None, plot=True, folder='../workspace/',
#                              save_img=False, save_npy=False)

t1 = datetime.now()
data.normalize_data(data_key=data_key, norm_data_key=None,
                    t_start=None, t_end=None, p_outlier=0.01, stop_thres=(0.1, 0.01),
                    window_s=60, step_s=5, fps=64)
t2 = datetime.now()
print('Spend time: {0} s'.format((t2-t1).total_seconds()))

#
data.plot_detected_veh_in_period(data_key='norm_'+data_key,
                             t_start= periods[dataset][0],
                             t_end=periods[dataset][1],
                             cbar_limit=(0,4), option='vec', nan_thres_p=0.9, det_thres=20,
                             plot=True, folder='../workspace/', save_img=False, save_npy=False)

# data.det_veh_in_img(data_key='norm_'+data_key, t_start=periods[dataset][0], t_end=periods[dataset][1],
#                     option='vec')

# img = data.get_heat_img_in_period(dataset=dataset, data_key='norm_'+data_key,
#                                 t_start=periods[dataset][0]+timedelta(seconds=35),
#                                 t_end=periods[dataset][0]+timedelta(seconds=65),
#                                 cbar_limit=[0, 5], option='vec',
#                                 nan_thres=None, plot=False,
#                                   dur=1, folder='../figs/hough_realdata/')


# ================================================================================
# Check nonlinear transformation
# ================================================================================
# sample_veh = np.load('../workspace/20161013_213255_441458.npy')
# data.nonlinear_transform(sample_veh[()], ratio=6)



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