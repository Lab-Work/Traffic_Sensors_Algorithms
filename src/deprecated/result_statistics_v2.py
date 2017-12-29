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
    # Jun 08
    if True:
        # Load the detection results
        folder = 'Jun08_2017'
        est_file = '../workspace/{0}/figs/speed/v2_6/detected_vehs.txt'.format(folder)
        true_file = '../workspace/{0}/data_convexlog_v2_cleaned.npy'.format(folder)

        t_start = str2time('2017-06-08 21:39:00.001464')
        t_end = str2time('2017-06-08 22:20:37.738293')
        init_t = str2time('2017-06-08 21:32:18.252811')

        offset = -0.28
        drift_ratio = 1860.5/1864.0

    # Jun 9
    if False:
        # Load the detection results
        folder = 'Jun09_2017'
        est_file = '../workspace/{0}/figs/speed/v2_3/detected_vehs_cleaned.txt'.format(folder)
        true_file = '../workspace/{0}/data_convexlog_v2_cleaned.npy'.format(folder)

        t_start = str2time('2017-06-09 19:09:00.009011')
        t_end = str2time('2017-06-09 20:39:30.905936')
        init_t = str2time('2017-06-09 19:09:00.0')

        offset = -0.66
        drift_ratio = 1860.5/1864.0

    vehs = load_det_results(est_file)
    compute_ultra_FN_ratio(vehs)

    speed_range = (0, 50)
    vehs = clean_det_data(vehs, speed_range)

    true_vehs = load_true_results(true_file, init_t, t_start, t_end, offset=offset, drift_ratio=drift_ratio)

    true_vehs = clean_true_data(true_vehs)

    # ====================================================
    # Just plot the distribution
    if False:
        _valid_idx = (vehs[:,4]==False)
        # true all VS est all
        # plot_distance_dist([true_vehs[:,3], vehs[:,2]], ['true all', 'est all'], title='Distance: true all vs. est all')
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
        dt = 1.5
        matched_vehs, fp_vehs = match_vehs(vehs, true_vehs, dt)
        num_fp = len(fp_vehs)

        # ====================================================
        # plot the distributions
        _idx = ~np.isnan(matched_vehs[:,0])

        for i, v in enumerate(matched_vehs):
            if (v[3] - v[2]) <= -10:
                print('Outliers: ')
                print('        {0}: true_d {1:.2f}, est_d {2:.2f}, true_v {3:.2f}, est_v {4:.2f}'.format(i, v[0], v[1],
                                                                                                         v[2], v[3]))

        num_fn = np.sum(np.isnan(matched_vehs[:,0]))
        plot_vehs = matched_vehs[_idx, :]
        _valid_idx = (plot_vehs[:,4] == True)

        print('\nDetection Statistics:')
        print('     True total: {0}'.format(len(true_vehs)))
        print('      TP     FP      FN')
        print('     {0}     {1}     {2}'.format(len(plot_vehs), num_fp, num_fn ))

        e_rms, e_valid_rms = compute_speed_rms(matched_vehs)
        print('\nSpeed Estimation Statistics:')
        print('     RMSE (mph)         all data: {0}'.format(e_rms))
        print('     RMSE (mph) with valid ultra: {0}'.format(e_valid_rms))

        # distance
        # true all VS est all
        # plot_distance_dist([plot_vehs[:,0], plot_vehs[:,1]], ['true all', 'est all'], title='Distance: true all vs. est all')
        # # est all VS est valid
        # plot_distance_dist([plot_vehs[:,1], plot_vehs[_valid_idx,1]], ['est all', 'est valid'],
        #                    title='Distance: est all vs. est valid')
        # # true valid VS est valid
        # plot_distance_dist([plot_vehs[_valid_idx,0], plot_vehs[_valid_idx,1]], ['true valid', 'est valid'],
        #                    title='Distance: true valid vs. est valid')

        # true all VS est all
        # plot_speed_dist([plot_vehs[:,2], plot_vehs[:,3]], ['true all', 'est all'], title='Speed: true all vs. est all')
        # # est all VS est valid
        # plot_speed_dist([plot_vehs[:,3], plot_vehs[_valid_idx,3]], ['est all', 'est valid'],
        #                 title='Speed: est all vs. est valid')
        # # true valid VS est valid
        # plot_speed_dist([plot_vehs[_valid_idx,2], plot_vehs[_valid_idx,3]], ['true valid', 'est valid'],
        #                 title='Speed: true valid vs. est valid')

        est_err = plot_vehs[:,3] - plot_vehs[:,2]
        plot_speed_dist([est_err], ['Speed estimation error'], title='Test 1 speed estimation error')
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
    dist_lim = [3.5, 8.0]

    mean_dist = 6.0
    for veh in vehs:

        # First, update the speed using historical median distance; and update historical median if reading is available
        if veh[2] <= dist_lim[0] or veh[2] >= dist_lim[1]:
            veh[3] = veh[3]*mean_dist/veh[2]
            veh[2] = mean_dist
        else:
            dists.append(veh[2])
            mean_dist = np.median(dists)

        # Second, cap the speed in the range
        if veh[3] < speed_range[0]:
            veh[3] = speed_range[0]
            veh[4] = False
        elif veh[3] > speed_range[1]:
            veh[3] = speed_range[1]
            veh[4] = False

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


def load_true_results(true_file, init_t, t_start=None, t_end=None, offset=0.0, drift_ratio=1.0):
    """
    This function loads the true result file (npy)
         start time, end time, speed (m/s), distance (m), image speed (px/frame), image distance (px)
    :param true_file: str
    :return:
    """
    true_vehs = np.load(true_file).tolist()

    cleaned_true_veh = []
    # convert the first column into datetime and the speed to mph
    for true_v in true_vehs:
        # fix the drift
        true_v[0] = init_t + timedelta(seconds=true_v[0]*drift_ratio + offset)
        true_v[1] = init_t + timedelta(seconds=true_v[1]*drift_ratio + offset)
        true_v[2] *= 2.23694

        # cleaned_true_veh.append(true_v)
        # remove the nan values
        if not np.isnan(true_v[2]) and \
                (t_start is None or true_v[0]>=t_start) and \
                (t_end is None or true_v[1]<=t_end):
            cleaned_true_veh.append(true_v)

    cleaned_true_veh = np.asarray(cleaned_true_veh)

    return cleaned_true_veh


def compute_ultra_FN_ratio(vehs):
    """
    This function computes the false negative ratio of the ultrasonic sensor.
    :param vehs: the vehs ndarray
    :return:
    """
    fn_ratio = sum(vehs[:,4]==False)/float(len(vehs))
    print('\n\nTotal number of detected vehicles: {0}'.format(len(vehs)))
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

        cur_min_dt = np.inf
        cur_matched_idx = 0
        # first find its closest vehicle
        for i in range(len(true_vehs)):
            _dt = abs( (_t_m - _true_t_m[i]).total_seconds() )
            if _dt <= cur_min_dt:
                cur_min_dt = _dt
                cur_matched_idx = i

        if cur_min_dt <= dt:
            # found the match, then check if it is duplicated detection
            true_v = true_vehs[cur_matched_idx]

            if not np.isnan(matched_vehs[cur_matched_idx][5]):
                # if the true vehicle has already been matched
                print('WARNING: Two matches for true vehicle: {0} ~ {1},'.format(true_v[0],true_v[1]) +
                      ' {0:.2f} m ; {1:.2f} mph'.format(true_v[3],true_v[2]))
                old_i = int(matched_vehs[cur_matched_idx][5])

                print('                            Old match: {0} ~ {1},'.format(vehs[old_i][0],vehs[old_i][1]) +
                      ' {0:.2f} m ; {1:.2f} mph'.format(vehs[old_i][2],vehs[old_i][3]))
                print('                            New match: {0} ~ {1},'.format(v[0],v[1]) +
                      ' {0:.2f} m ; {1:.2f} mph'.format(v[2],v[3]))

                # compare which one is closer
                old_dt =  abs((vehs[old_i][0] + (vehs[old_i][1] - vehs[old_i][0])/2 -
                               _true_t_m[cur_matched_idx]).total_seconds())
                if cur_min_dt <= old_dt:
                    print ('                            New replaced Old.')
                    # replace old match, and append old to false positive
                    # match the vehicle, [true_dist, est_dist, true_speed, est_speed, valid, idx]
                    matched_vehs[cur_matched_idx][0], matched_vehs[cur_matched_idx][1] = true_v[3], v[2]
                    matched_vehs[cur_matched_idx][2], matched_vehs[cur_matched_idx][3] = true_v[2], v[3]
                    if v[4] is True:
                        matched_vehs[cur_matched_idx][4] = True
                    else:
                        matched_vehs[cur_matched_idx][4] = False
                    matched_vehs[cur_matched_idx][5] = v_idx

                    fp_vehs.append(vehs[old_i])
                else:
                    # otherwise, append to false positive
                    fp_vehs.append(v)

            else:
                # match the vehicle, [true_dist, est_dist, true_speed, est_speed, valid, idx]
                matched_vehs[cur_matched_idx][0], matched_vehs[cur_matched_idx][1] = true_v[3], v[2]
                matched_vehs[cur_matched_idx][2], matched_vehs[cur_matched_idx][3] = true_v[2], v[3]
                if v[4] is True:
                    matched_vehs[cur_matched_idx][4] = True
                else:
                    matched_vehs[cur_matched_idx][4] = False
                matched_vehs[cur_matched_idx][5] = v_idx
        else:
            # did not find a match, then as a false positive
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
    plt.text(mu-2*std, np.max(n)*1.1, text_str, fontsize=16)
    plt.ylim([0, np.max(n)*1.2])

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

        # text.append('{0} mean: {1:.2f}, std: {2:.2f}'.format(labels[i], mu, std))
        text.append('Mean: {0:.2f}\nStandard deviation: {1:.2f}'.format(mu, std))

    text_str = '\n'.join(text)
    plt.text(mu-3*std, np.max(n)*1.03, text_str, fontsize=30)
    # plt.text(6, 0.16, text_str, fontsize=16)
    plt.ylim(0, np.max(n)*1.2)

    # plt.legend()
    plt.xlabel('Speed (mph)', fontsize=30)
    plt.ylabel('Distribution', fontsize=30)
    plt.title(title, fontsize=34)
    plt.tick_params(axis='both', which='major', labelsize=28)
    plt.draw()




if __name__ == '__main__':
    main()
