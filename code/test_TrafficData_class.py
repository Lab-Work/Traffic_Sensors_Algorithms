__author__ = 'Yanning'


from TrafficDataClasses import *

file = 'cleaned_data.csv'

pir_data = TrafficData()

pir_data.read_data_file(file)

# pir_data.subtract_PIR_background(40, 'cleaned_data.csv')

pir_data.plot_time_series_for_pixel(t_start_str='20150903_15:01:15_840141',
                                    t_end_str='20150903_15:02:15_840141',
                                    pixel_list=[('pir_4x48', [(1,1),(1,23),(1,47)])],
                                    data_option='raw')

# pir_data.plot_histogram_for_pixel(t_start_str='20150903_15:01:15_840141',
#                                   t_end_str='20150903_15:06:15_840141',
#                                   pixel_list=[('pir_4x48', [(1,1),(1,23),(1,47)])])

pir_data.plot_heat_map_in_period(t_start_str='20150903_15:01:15_840141',
                                 t_end_str='20150903_15:02:15_840141',
                                 T_min=-5, T_max=7, data_option='raw')


plt.show()

