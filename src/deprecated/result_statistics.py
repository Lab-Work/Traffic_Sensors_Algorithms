import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from datetime import datetime
from datetime import timedelta
import itertools


"""
This script is used to analyze the statistics of the speed estimation result.

The data columns are:
t_in, t_out, dist (m), speed (mph), valid?
The true data columns are:
start time, end time, speed (m/s), distance (m), image speed (px/frame), image distance (px)
"""

def main():

    # ====================================================
    # Load the detection results
    folder = 'Jun08_2017'
    result_file = '../workspace/{0}/figs/speed/v2_2/detected_vehs.txt'.format(folder)
    vehs = load_det_results(result_file)

    t_start = str2time('2017-06-08 21:39:00.001464')
    t_end = str2time('2017-06-08 22:20:37.738293')

    compute_ultra_FN_ratio(vehs)

    speed_range = (0, 50)
    vehs = clean_det_data(vehs, speed_range)

    # ====================================================
    # load true file
    true_file = '../workspace/{0}/data_convexlog_v2.npy'.format(folder)
    init_t = str2time('2017-06-08 21:32:18.252811')
    true_vehs = load_true_results(true_file, init_t, t_start, t_end)

    true_vehs = clean_true_data(true_vehs)

    # ====================================================
    # Just plot the distribution
    if False:
        _valid_idx = (vehs[:,4]==False)
        # true all VS est all
        plot_distance_dist([true_vehs[:,3], vehs[:,2]], ['true all', 'est all'], title='Distance: true all vs. est all')
        # # est all VS est valid
        # plot_distance_dist([vehs[:,2], vehs[_valid_idx,2]], ['est all', 'est valid'],
        #                    title='Distance: est all vs. est valid')
        # # true valid VS est valid
        # plot_distance_dist([true_vehs[:,3], vehs[_valid_idx,2]], ['true valid', 'est valid'],
        #                    title='Distance: true valid vs. est valid')
        #
        # # true all VS est all
        plot_speed_dist([true_vehs[:,2]], ['true all'], title='Speed: true all vs. est all')
        # plot_speed_dist([true_vehs[:,2], vehs[:,3]], ['true all', 'est all'], title='Speed: true all vs. est all')
        # # est all VS est valid
        # plot_speed_dist([vehs[:,3], vehs[_valid_idx,3]], ['est all', 'est valid'],
        #                 title='Speed: est all vs. est valid')
        # # true valid VS est valid
        # plot_speed_dist([true_vehs[:,2], vehs[_valid_idx,3]], ['true valid', 'est valid'],
        #                 title='Speed: true valid vs. est valid')

    # ====================================================
    # match vehicles and compute error
    if True:
        dt = 0.5
        matched_vehs, fp_vehs = match_vehs(vehs, true_vehs, dt)
        num_fp = len(fp_vehs)

        # ====================================================
        # plot the distributions
        _idx = ~np.isnan(matched_vehs[:,0])
        num_fn = np.sum(np.isnan(matched_vehs[:,0]))
        plot_vehs = matched_vehs[_idx, :]
        _valid_idx = (plot_vehs[:,4] == True)

        print('\n\n\n')
        print('Detection Statistics:')
        print('     True total: {0}'.format(len(true_vehs)))
        print('      TP     FP      FN')
        print('     {0}     {1}     {2}'.format(len(plot_vehs), num_fp, num_fn ))

        e_rms, e_valid_rms = compute_speed_rms(matched_vehs)
        print('\nSpeed Estimation Statistics:')
        print('     RMSE (mph)         all data: {0}'.format(e_rms))
        print('     RMSE (mph) with valid ultra: {0}'.format(e_valid_rms))

        # distance
        # true all VS est all
        plot_distance_dist([plot_vehs[:,0], plot_vehs[:,1]], ['true all', 'est all'], title='Distance: true all vs. est all')
        # # est all VS est valid
        # plot_distance_dist([plot_vehs[:,1], plot_vehs[_valid_idx,1]], ['est all', 'est valid'],
        #                    title='Distance: est all vs. est valid')
        # # true valid VS est valid
        # plot_distance_dist([plot_vehs[_valid_idx,0], plot_vehs[_valid_idx,1]], ['true valid', 'est valid'],
        #                    title='Distance: true valid vs. est valid')

        # true all VS est all
        plot_speed_dist([plot_vehs[:,2], plot_vehs[:,3]], ['true all', 'est all'], title='Speed: true all vs. est all')
        # # est all VS est valid
        # plot_speed_dist([plot_vehs[:,3], plot_vehs[_valid_idx,3]], ['est all', 'est valid'],
        #                 title='Speed: est all vs. est valid')
        # # true valid VS est valid
        # plot_speed_dist([plot_vehs[_valid_idx,2], plot_vehs[_valid_idx,3]], ['true valid', 'est valid'],
        #                 title='Speed: true valid vs. est valid')

        # est_err = plot_vehs[:,3] - plot_vehs[:,2]
        # plot_speed_dist([est_err], ['Speed error'], title='Error: true all vs. est all')
        # est_err = plot_vehs[_valid_idx,3] - plot_vehs[_valid_idx,2]
        # plot_speed_dist([est_err], ['Speed error'], title='Error: true valid vs. est valid')

        # ratio = plot_vehs[:,3]/plot_vehs[:,2]
        # plot_speed_dist([ratio], ['Speed ratio'], title='ratio: est all / true all')


    plt.show()



def time2str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def str2time(dt_str):
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")

def str2bool(str):
    if str == 'True' or str == 'true':
        return True
    else:
        return False

def clean_det_data(vehs, speed_range):
    """
    This function cleans the detection data:
        - Make sure speed is in speed range
        - use the historical median distance to compute the speed if ultrasonic reading is not available
        - Put threshold to check if ultrasonic sensor reading is on the other lane or not.
    :param vehs: the loaded vehs:  [t_in, t_out, dist(m), speed(mph), est_dist(B)]
    :param speed_range: tuple, speeds in mph
    :return:
    """
    dists = []
    mean_dist = 4.0
    for veh in vehs:

        # First, update the speed using historical median distance; and update historical median if reading is available
        if veh[4] is True or veh[2]>= 8.0:
            veh[3] = veh[3]*mean_dist/veh[2]
            veh[2] = mean_dist
        else:
            dists.append(veh[2])
            mean_dist = np.median(dists)

        # Second, cap the speed in the range
        if veh[3] < speed_range[0]:
            veh[3] = speed_range[0]
            veh[4] = True
        elif veh[3] > speed_range[1]:
            veh[3] = speed_range[1]
            veh[4] = True

    return vehs


def clean_true_data(true_vehs):
    """
    There are some -inf data in the true_veh data, clean those rows.
    :param true_vehs:
    :return:
    """
    cleaned_vehs = []
    for v in true_vehs:
        if not np.isinf(v[2]):
            cleaned_vehs.append(v)

    return np.asarray(cleaned_vehs)


def load_det_results(result_file):
    """
    This function loads the detection results:
        [t_in, t_out, dist(m), speed(mph), est_dist(B)]
    :param result_file:
    :return:
    """
    vehs = []
    with open(result_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split(',')
            vehs.append([str2time(items[0]), str2time(items[1]),
                         float(items[2]), float(items[3]), str2bool(items[4])])

    vehs = np.asarray(vehs)
    return vehs


def load_true_results(true_file, init_t, t_start=None, t_end=None):
    """
    This function loads the true result file (npy)
         start time, end time, speed (m/s), distance (m), image speed (px/frame), image distance (px)
    :param true_file: str
    :return:
    """
    true_vehs = np.load(true_file).tolist()

    print('Cleaning the true vehicles data from {0} vehicles to'.format(len(true_vehs)))

    cleaned_true_veh = []
    # convert the first column into datetime and the speed to mph
    for true_v in true_vehs:
        # the video data is a bit drifting. 14 seconds over 2 hrs
        # There may be a time drift here
        # true_v[0] = init_t + timedelta(seconds=true_v[0]*7187.5/7200)
        # true_v[1] = init_t + timedelta(seconds=true_v[1]*7187.5/7200)
        drift_cor = 1860.5/1864.0
        offset = 0.28
        true_v[0] = init_t + timedelta(seconds=true_v[0]*drift_cor - offset)
        true_v[1] = init_t + timedelta(seconds=true_v[1]*drift_cor - offset)
        true_v[2] *= 2.24

        # cleaned_true_veh.append(true_v)
        # remove the nan values
        if not np.isnan(true_v[2]) and \
                (t_start is None or true_v[0]>=t_start) and \
                (t_end is None or true_v[1]<=t_end):
            cleaned_true_veh.append(true_v)

    cleaned_true_veh = np.asarray(cleaned_true_veh)

    print('                           clean {0} vehicles'.format(len(cleaned_true_veh)))

    return cleaned_true_veh


def compute_ultra_FN_ratio(vehs):
    """
    This function computes the false negative ratio of the ultrasonic sensor.
    :param vehs: the vehs ndarray
    :return:
    """
    fn_ratio = sum(vehs[:,4]==True)/float(len(vehs))
    print('Total numebr of detected vehicles: {0}'.format(len(vehs)))
    print('False negative ratio of ultrasonic sensor: {0}'.format(fn_ratio))


def match_vehs(vehs, true_vehs, dt):
    """
    This function computes the speed estimation error:
        - The detected vehicle matches its closest true vehicle that (t_in + t_out)/2 differs less than dt
    :param vehs: detected vehicles  [t_in, t_out, dist(m), speed(mph), est_dist(B)]
    :param true_vehs: true vehicles [t_in, t_out, speed (mph), distance (m), image speed (px/frame), image distance (px)]
    :return: mathed vehs: [true_dist, est_dist, true_speed, est_speed, valid, idx] in same dimension as true_vehs
        entry valid is True if the speed is not estimated from median distance or capped from speed range
        entry idx is the index for the matched vehicle in vehs
    """
    matched_vehs = np.ones((len(true_vehs),6))*np.nan

    # this list saves the unmatched vehicles in vehs
    fp_vehs = []

    # compute the temparary (t_in+t_out)/2 for true
    _true_t_m = [v[0] + (v[1]-v[0])/2 for v in true_vehs]

    # check each vehicle in vehs
    for v_idx, v in enumerate(vehs):
        _t_m = v[0] + (v[1]-v[0])/2

        flag = False
        for i, t_v in enumerate(true_vehs):
            if abs( (_t_m - _true_t_m[i]).total_seconds() ) <= dt:

                if not np.isnan(matched_vehs[i][5]):
                    # if the true vehicle has already been matched
                    print('WARNING: Two matches for true vehicle: {0} ~ {1},'.format(t_v[0],t_v[1]) +
                          ' {0:.2f} m ; {1:.2f} mph'.format(t_v[3],t_v[2]))
                    _i = matched_vehs[i][5]
                    print('_i: {0}'.format(_i))

                    print('                            Match one: {0} ~ {1},'.format(vehs[_i][0],vehs[_i][1]) +
                          ' {0:.2f} m ; {1:.2f} mph'.format(vehs[_i][3],vehs[_i][2]))
                    print('                            Match one: {0} ~ {1},'.format(v[0],v[1]) +
                          ' {0:.2f} m ; {1:.2f} mph'.format(v[2],v[3]))
                else:
                    # match the vehicle, [true_dist, est_dist, true_speed, est_speed, valid, idx]
                    matched_vehs[i][0], matched_vehs[i][1] = t_v[3], v[2]
                    matched_vehs[i][2], matched_vehs[i][3] = t_v[2], v[3]
                    if v[4] is True:
                        matched_vehs[i][4] = False
                    else:
                        matched_vehs[i][4] = True
                    matched_vehs[i][5] = v_idx
                    flag = True
                    break

        # check if matched
        if flag is False:
            fp_vehs.append(v)

    return matched_vehs, fp_vehs


def compute_speed_rms(matched_vehs):
    """
    This function computes the RMSE of the speeds
    :param matched_vehs: [true_dist, est_dist, true_speed, est_speed, valid, idx]
    :return: float, err (mph)
    """
    _idx = ~np.isnan(matched_vehs[:,0])


    e_rms = np.sqrt(np.mean(np.power(matched_vehs[_idx, 2] - matched_vehs[_idx, 3], 2)))

    _tmp_vehs = matched_vehs[_idx,:]

    _valid_idx = (_tmp_vehs[:,4] == True)

    e_valid_rms = np.sqrt(np.mean(np.power(_tmp_vehs[_valid_idx, 2] - _tmp_vehs[_valid_idx, 3], 2)))

    return e_rms, e_valid_rms


def plot_distance_dist(l_dists, labels, title='Distance distribution'):
    """
    This function plots the distance distribution of the list of distances with labels
    :param l_dists: [[dists], [dists]]
    :param labels: ['true', 'est']
    :return:
    """

    plt.figure(figsize=(10,10))
    colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
    text = []
    for i, dists in enumerate(l_dists):
        c = next(colors)
        mu, std = np.mean(dists), np.std(dists)
        n, bins, patches = plt.hist(dists, 50, normed=1, facecolor=c, alpha=0.75, label=labels[i])
        plt.plot(bins, mlab.normpdf(bins, mu, std), color=c, linestyle='--',
                 linewidth=2)

        text.append('{0} mean: {1:.2f}, std: {2:.2f}'.format(labels[i], mu, std))

    text_str = '\n'.join(text)
    plt.text(mu*1.05, np.max(n)*0.7, text_str, fontsize=16)

    plt.legend()
    plt.xlabel('Distance (m)', fontsize=16)
    plt.ylabel('Distribution', fontsize=16)
    plt.title(title, fontsize=20)
    plt.draw()


def plot_speed_dist(l_speeds, labels, title='Speed distribution'):
    """
    This function plots the speeds distribution of the list of distances with labels
    :param l_speeds: [[speeds], [speeds]]
    :param labels: ['true', 'est']
    :return:
    """

    plt.figure(figsize=(10,10))
    colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
    text = []
    for i, speeds in enumerate(l_speeds):
        c = next(colors)
        mu, std = np.mean(speeds), np.std(speeds)
        n, bins, patches = plt.hist(speeds, 50, normed=1, facecolor=c, alpha=0.75, label=labels[i])
        plt.plot(bins, mlab.normpdf(bins, mu, std), color=c, linestyle='--',
                 linewidth=2)

        text.append('{0} mean: {1:.2f}, std: {2:.2f}'.format(labels[i], mu, std))

    text_str = '\n'.join(text)
    plt.text(5, np.max(n)*1.0, text_str, fontsize=16)
    # plt.text(6, 0.16, text_str, fontsize=16)

    plt.legend()
    plt.xlabel('Speed (mph)', fontsize=16)
    plt.ylabel('Distribution', fontsize=16)
    plt.title(title, fontsize=20)
    plt.draw()




# # ========================================================================
# # Compute the number of false positives
# # true_v [6] is (t1_in+t2_out)/2, true_v[7] estimated speed, true_v[8] estimated distance
# if False:
#     true_vehs = np.hstack([true_vehs, np.ones((len(true_vehs),3))*np.nan])
#     for true_v in true_vehs:
#         true_v[6] = true_v[0] + (true_v[1]-true_v[0])/2
#
#     # vehs[5] is (t_in + t_out)/2
#     fp_vehs = []
#     vehs = np.hstack([vehs, np.ones((len(vehs), 1))*np.nan])
#     for v in vehs:
#         v[5] = v[0] + (v[1]-v[0])/2
#
#         if v[4] is False:
#             # find the true vehs that within 1 s bound
#             flag = False
#             for true_v in true_vehs:
#                 # print (true_v[6]-v[5]).total_seconds()
#                 if abs((true_v[6]-v[5]).total_seconds()) <=0.5:
#                     # found the true veh, save the estimated speed
#                     if np.isnan(true_v[7]):
#                         true_v[7] = v[3]
#                         true_v[8] = v[2]
#                         flag = True
#                         break
#
#             if flag is False:
#                 fp_vehs.append(v)
#
#     num_fp = len(fp_vehs)
#
#     # compute the speed estimation errors RMSE
#     err = []
#     for true_v in true_vehs:
#             if not np.isnan(true_v[7]):
#                 err.append(true_v[7] - true_v[2])
#
#     err = np.asarray(err)
#     # compute rmse
#     e_rms = np.sqrt(np.mean( err**2 ))


# # ========================================================================
# # plot all detected vehicles on both lanes and the true vehicle on the closer lane
# if False:
#     fig = plt.figure(figsize=(15,8))
#     ax = fig.add_axes([0.1, 0.15, 0.82, 0.75])
#
#     # plot the true speed estimation
#     for true_v in true_vehs:
#         ax.plot([true_v[0], true_v[1]], [true_v[2], true_v[2]], linewidth=2, linestyle='--', color='g', label='true')
#         ax.axvspan(true_v[0], true_v[1], facecolor='g', edgecolor='g', alpha=0.5)
#
#     # plot the detected vehicle speed in the closer lane and further lane
#     for est_v in vehs:
#         if est_v[4] is False:
#             # with ultrasonic readings
#             ax.plot([est_v[0], est_v[1]], [est_v[3], est_v[3]], linewidth=2, linestyle='--', color='b', label='closer_est')
#         else:
#             # no ultrasonic readings
#             # ax.plot([est_v[0], est_v[1]], [est_v[3], est_v[3]], linewidth=2, linestyle='--', color='r', label='further_est')
#             pass
#
#     # plot false positives
#     for fp_v in fp_vehs:
#         ax.plot([fp_v[0], fp_v[1]], [fp_v[2], fp_v[2]], linewidth=2, linestyle='-', color='r', label='FP')
#
#     ax.set_title('Detection results', fontsize=18)
#     ax.set_xlabel('Time', fontsize=14)
#     ax.set_ylabel('Speed (mph)', fontsize=14)
#     # plt.legend()
#
#     plt.draw()
#
# # ========================================================================
# # plot the speed esitmation error distribution
#
#
# # ========================================================================
# # plot the distance distribution
# if False:
#     # true distance
#     true_dist = []
#     est_dist = []
#
#     for true_v in true_vehs:
#         if not np.isnan(true_v[7]):
#             true_dist.append(true_v[3])
#             est_dist.append((true_v[8]))
#
#     plt.figure(figsize=(10,10))
#     n, bins, patches = plt.hist(true_dist, 50, normed=1, facecolor='green', alpha=0.75, label='true')
#     plt.plot(bins, mlab.normpdf(bins, np.mean(true_dist), np.std(true_dist)), color='g', linestyle='--',
#              label='true', linewidth=2)
#
#     n, bins, patches = plt.hist(est_dist, 50, normed=1, facecolor='blue', alpha=0.75, label='est')
#     plt.plot(bins, mlab.normpdf(bins, np.mean(est_dist), np.std(est_dist)), color='b', linestyle='--',
#              label='est', linewidth=2)
#
#     plt.text(2, 1.5, 'true mean: {0:.2f}, std: {1:.2f}\nest mean: {2:.2f}, std: {3:.2f}'.format(np.mean(true_dist),
#                                                                                                 np.std(true_dist),
#              np.mean(est_dist), np.std(est_dist)))
#
#     plt.legend()
#     plt.xlabel('Distance (m)', fontsize=14)
#
# # ========================================================================
# # plot the distribution of speeds with all valid distance readings from the ultrasonic sensor
# if True:
#     valid_idx = (vehs[:,4] == False)
#     invalid_idx = (vehs[:,4] == True)
#
#     plt.figure(figsize=(10,10))
#     # plot the valid speed distribution
#     n, bins, patches = plt.hist(vehs[valid_idx, 3], 50, normed=1, facecolor='green', alpha=0.75)
#     plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[valid_idx, 3]), np.std(vehs[valid_idx, 3])), color='r', linestyle='--',
#              label='valid ultra', linewidth=2)
#
#     # plot the invalid speed distribution
#     n, bins, patches = plt.hist(vehs[invalid_idx, 3], 50, normed=1, facecolor='blue', alpha=0.75)
#     plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[invalid_idx, 3]), np.std(vehs[invalid_idx, 3])), color='k', linestyle='--',
#              label='invalid ultra', linewidth=2)
#     plt.xlabel('Speed (mph)', fontsize=14)
#     plt.title('Speed distribution', fontsize=16)
#     plt.legend()
#
#
# # plot all speed distribution
# if False:
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_axes([0.1, 0.15, 0.82, 0.75])
#     n, bins, patches = plt.hist(vehs[:, 3], 50, normed=1, facecolor='green', alpha=0.75)
#     # plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[valid_idx, 3]), np.std(vehs[valid_idx, 3])), color='r', linestyle='--',
#     #          linewidth=2)
#     plt.title('Distribution of estimated speeds', fontsize=40)
#     plt.xlabel('Speed (mph)', fontsize=36)
#     ax.tick_params(axis='both', which='major', labelsize=24)
#
#     # plot the speed
#     fig = plt.figure(figsize=(15,8))
#     ax = fig.add_axes([0.1, 0.15, 0.82, 0.75])
#     plt.plot(vehs[:,3], color='g', linewidth=2)
#     plt.xlabel('Vehicle index', fontsize=36)
#     plt.ylabel('Speed (mph)', fontsize=36)
#     plt.title('Speed estiamtion accuracy', fontsize=40)
#     ax.tick_params(axis='both', which='major', labelsize=24)
#
# # ========================================================================
# # plot the distribution of the distance
# if True:
#     plt.figure(figsize=(10,10))
#     n, bins, patches = plt.hist(vehs[:, 2], 50, normed=1, facecolor='green', alpha=0.75)
#     plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[valid_idx, 3]), np.std(vehs[valid_idx, 3])), color='r', linestyle='--',
#              label='distance', linewidth=2)
#     plt.xlabel('Distance (m)', fontsize=14)
#     plt.title('Distance distribution', fontsize=16)
#
#     plt.show()
#
#
#
#
# plt.show()


if __name__ == '__main__':
    main()
