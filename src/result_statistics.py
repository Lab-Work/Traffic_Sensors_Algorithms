import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from datetime import datetime
from datetime import timedelta


"""
This script is used to analyze the statistics of the speed estimation result.

The data columns are:
t_in, t_out, dist (m), speed (mph), estimated_dist
"""

def time2str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def str2time(dt_str):
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")

def str2bool(str):
    if str == 'True' or str == 'true':
        return True
    else:
        return False

# ========================================================================
# Read in the estimated result
# t_in, t_out, dist, speed, est_dist
result_file = '../workspace/0509_2017/figs/speed/complete_linear_residual_final/detected_vehs.txt'
vehs = []
with open(result_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split(',')
        vehs.append([str2time(items[0]), str2time(items[1]),
                     float(items[2]), float(items[3]), str2bool(items[4])])

vehs = np.asarray(vehs)

# ========================================================================
# clean the vehs data
# if the speed is estimated and the distance is estimated, then use updated estimated speed 5.5 m.
for veh in vehs:
    # those overestimated speeds may be on the closer lane
    if veh[4] == True and veh[3] >= 60:
        veh[3] = veh[3]*5.5/veh[2]
        veh[4] = False  # pretty confident that the vehicle is on the closer lane

    if veh[2] > 7.5:
        # further than 7.5 meter, on the other lane
        veh[4] = True
# Now veh[4] means: the vehicle is on the further lane

# ========================================================================
# Read in the ground truth
# start time, end time, speed (m/s), distance (m), image speed (px/frame), image distance (px)
# The zero time in the data corresponds to actual machine time 2017-05-09 20:05:19
true_file = '../datasets/0509_2017/050917_V2_convexlog.npy'
true_vehs = np.load(true_file).tolist()
# convert the first colume into datetime and the speed to mph
init_t = str2time('2017-05-09 20:05:17.5')
for true_v in true_vehs:
    # the video data is a bit drifting. 14 seconds over 2 hrs
    true_v[0] = init_t + timedelta(seconds=true_v[0]*7187.5/7200)
    true_v[1] = init_t + timedelta(seconds=true_v[1]*7187.5/7200)
    true_v[2] *= 2.24

true_vehs = np.asarray(true_vehs)

# ========================================================================
# print the FN ratio
fn_ratio = sum(vehs[:,4]==True)/float(len(vehs))
print('Total numebr of detected vehicles: {0}'.format(len(vehs)))
print('False negative ratio of ultrasonic sensor: {0}'.format(fn_ratio))

# ========================================================================
# Compute the number of false positives
# true_v [6] is (t1_in+t2_out)/2, true_v[7] estimated speed, true_v[8] estimated distance
true_vehs = np.hstack([true_vehs, np.ones((len(true_vehs),3))*np.nan])
for true_v in true_vehs:
    true_v[6] = true_v[0] + (true_v[1]-true_v[0])/2

# vehs[5] is (t_in + t_out)/2
fp_vehs = []
vehs = np.hstack([vehs, np.ones((len(vehs), 1))*np.nan])
for v in vehs:
    v[5] = v[0] + (v[1]-v[0])/2

    if v[4] is False:
        # find the true vehs that within 1 s bound
        flag = False
        for true_v in true_vehs:
            # print (true_v[6]-v[5]).total_seconds()
            if abs((true_v[6]-v[5]).total_seconds()) <=0.5:
                # found the true veh, save the estimated speed
                if np.isnan(true_v[7]):
                    true_v[7] = v[3]
                    true_v[8] = v[2]
                    flag = True
                    break

        if flag is False:
            fp_vehs.append(v)

num_fp = len(fp_vehs)

# compute the speed estimation errors RMSE
err = []
for true_v in true_vehs:
        if not np.isnan(true_v[7]):
            err.append(true_v[7] - true_v[2])

err = np.asarray(err)
# compute rmse
e_rms = np.sqrt(np.mean( err**2 ))

# ========================================================================
# print some statistics
num_det = sum(vehs[:,4] == False)
print('\n\n\n')
print('Detection Statistics:')
print('     True total on closer lane: {0}'.format(len(true_vehs)))
print('     Detected on closer lane: {0} out of {1} for two lanes'.format(sum(vehs[:,4] == False), len(vehs)))
print('      TP     FP      FN')
print('     {0}     {1}     {2}'.format(num_det - num_fp, num_fp, len(true_vehs) + num_fp - num_det  ))

print('\nSpeed Estimation Statistics:')
print('     RMSE (mph): {0}'.format(e_rms))

# ========================================================================
# plot all detected vehicles on both lanes and the true vehicle on the closer lane
if True:
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1, 0.15, 0.82, 0.75])

    # plot the true speed estimation
    for true_v in true_vehs:
        ax.plot([true_v[0], true_v[1]], [true_v[2], true_v[2]], linewidth=2, linestyle='--', color='g', label='true')
        ax.axvspan(true_v[0], true_v[1], facecolor='g', edgecolor='g', alpha=0.5)

    # plot the detected vehicle speed in the closer lane and further lane
    for est_v in vehs:
        if est_v[4] is False:
            # with ultrasonic readings
            ax.plot([est_v[0], est_v[1]], [est_v[3], est_v[3]], linewidth=2, linestyle='--', color='b', label='closer_est')
        else:
            # no ultrasonic readings
            # ax.plot([est_v[0], est_v[1]], [est_v[3], est_v[3]], linewidth=2, linestyle='--', color='r', label='further_est')
            pass

    # plot false positives
    for fp_v in fp_vehs:
        ax.plot([fp_v[0], fp_v[1]], [fp_v[2], fp_v[2]], linewidth=2, linestyle='-', color='r', label='FP')

    ax.set_title('Detection results', fontsize=18)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Speed (mph)', fontsize=14)
    # plt.legend()

    plt.draw()

# ========================================================================
# plot the speed esitmation error distribution
if True:
    plt.figure(figsize=(10,10))
    # plot the valid speed distribution
    n, bins, patches = plt.hist(err, 50, normed=1, facecolor='green', alpha=0.75)

    # remove extreme errors
    _idx = (err>=-10) & (err <=20)
    err_clean = err[_idx]

    plt.plot(bins, mlab.normpdf(bins, np.mean(err_clean), np.std(err_clean)), color='r', linestyle='--',
             label='error distribution', linewidth=2)

    plt.text(-20, 0.08, 'mean: {0}\nstd:{1}'.format(np.mean(err_clean), np.std(err_clean)))

    plt.title('Distribution of estimated speed error', fontsize=18)
    plt.xlabel('Speed error (mph)', fontsize=14)
    plt.draw()


# ========================================================================
# plot the distance distribution
if True:
    # true distance
    true_dist = []
    est_dist = []

    for true_v in true_vehs:
        if not np.isnan(true_v[7]):
            true_dist.append(true_v[3])
            est_dist.append((true_v[8]))

    plt.figure(figsize=(10,10))
    n, bins, patches = plt.hist(true_dist, 50, normed=1, facecolor='green', alpha=0.75, label='true')
    plt.plot(bins, mlab.normpdf(bins, np.mean(true_dist), np.std(true_dist)), color='g', linestyle='--',
             label='true', linewidth=2)

    n, bins, patches = plt.hist(est_dist, 50, normed=1, facecolor='blue', alpha=0.75, label='est')
    plt.plot(bins, mlab.normpdf(bins, np.mean(est_dist), np.std(est_dist)), color='b', linestyle='--',
             label='est', linewidth=2)

    plt.text(2, 1.5, 'true mean: {0:.2f}, std: {1:.2f}\nest mean: {2:.2f}, std: {3:.2f}'.format(np.mean(true_dist),
                                                                                                np.std(true_dist),
             np.mean(est_dist), np.std(est_dist)))

    plt.legend()
    plt.xlabel('Distance (m)', fontsize=14)

# ========================================================================
# plot the distribution of speeds with all valid distance readings from the ultrasonic sensor
if False:
    valid_idx = (vehs[:,4] == False)
    invalid_idx = (vehs[:,4] == True)

    plt.figure(figsize=(10,10))
    # plot the valid speed distribution
    n, bins, patches = plt.hist(vehs[valid_idx, 3], 50, normed=1, facecolor='green', alpha=0.75)
    plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[valid_idx, 3]), np.std(vehs[valid_idx, 3])), color='r', linestyle='--',
             label='valid ultra', linewidth=2)

    # plot the invalid speed distribution
    n, bins, patches = plt.hist(vehs[invalid_idx, 3], 50, normed=1, facecolor='blue', alpha=0.75)
    plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[invalid_idx, 3]), np.std(vehs[invalid_idx, 3])), color='k', linestyle='--',
             label='invalid ultra', linewidth=2)
    plt.xlabel('Speed (mph)', fontsize=14)
    plt.title('Speed distribution', fontsize=16)
    plt.legend()


# plot all speed distribution
if False:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.1, 0.15, 0.82, 0.75])
    n, bins, patches = plt.hist(vehs[:, 3], 50, normed=1, facecolor='green', alpha=0.75)
    # plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[valid_idx, 3]), np.std(vehs[valid_idx, 3])), color='r', linestyle='--',
    #          linewidth=2)
    plt.title('Distribution of estimated speeds', fontsize=40)
    plt.xlabel('Speed (mph)', fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=24)

    # plot the speed
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1, 0.15, 0.82, 0.75])
    plt.plot(vehs[:,3], color='g', linewidth=2)
    plt.xlabel('Vehicle index', fontsize=36)
    plt.ylabel('Speed (mph)', fontsize=36)
    plt.title('Speed estiamtion accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=24)

# ========================================================================
# plot the distribution of the distance
if False:
    plt.figure(figsize=(10,10))
    n, bins, patches = plt.hist(vehs[:, 2], 50, normed=1, facecolor='green', alpha=0.75)
    plt.plot(bins, mlab.normpdf(bins, np.mean(vehs[valid_idx, 3]), np.std(vehs[valid_idx, 3])), color='r', linestyle='--',
             label='distance', linewidth=2)
    plt.xlabel('Distance (m)', fontsize=14)
    plt.title('Distance distribution', fontsize=16)

    plt.show()




plt.show()
