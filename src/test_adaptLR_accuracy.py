import sys, os
from os.path import exists

import itertools
import time, glob
from datetime import datetime
from datetime import timedelta
from contextlib import contextmanager
from copy import deepcopy
import imutils
from collections import OrderedDict
from ast import literal_eval as make_tuple

import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib.patches as patches

import numpy as np
import scipy
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

__author__ = 'Yanning Li'
__version__ = "1.0"
__email__ = 'yli171@illinois.edu'



"""
This script compares the LR results (s2 on May 30th) with the manually labeled true data (s2) to evaluated its accuracy.
"""

def main():

    # ===================================================================================
    # [start_time (dt), end_time (dt), speed (mph), distance (m), image speed (px/frame), image distance (px)]
    s2_true_file = '../workspace/0530_2017/labels_v21_post.npy'

    # each item is a dict:  ['t_in',  't_out', 'distance', 'speed', 'closer_lane']
    s2_lr_file = '../workspace/0530_2017/figs/speed/s2/v2_3/detected_vehs_post_comb_v2.npy'

    # -----------------------------------------------------------
    # Load true and save in array: [mean_time (dt), speed (mph)]
    _tmp = np.load(s2_true_file)
    s2_true = []
    for row in _tmp:
        s2_true.append( [row[0] + (row[1]-row[0])/2, row[2] ])
    s2_true.sort(key=lambda x:x[0])
    s2_true = np.asarray(s2_true)

    # -----------------------------------------------------------
    # Load detection result using adaptiveLR: [mean_time (dt), speed (mph)]
    _tmp = np.load(s2_lr_file)
    s2_lr = []
    for v in _tmp:
        if v['closer_lane'] is True:
            s2_lr.append( [ v['t_in'] + (v['t_out']-v['t_in'])/2, abs(v['speed']) ] )
    s2_lr.sort(key=lambda x:x[0])
    s2_lr = np.asarray(s2_lr)

    # ===================================================================================
    print('Loaded results:   true,   LR')
    print('                   {0},   {1}'.format(len(s2_true), len(s2_lr)))


    # -----------------------------------------------------------
    # match LR vs true
    dt = 0.5    # seconds
    # matches: a list of tuple [(item1, item2),... ], where first item from list1 and second from list2
    #          Each item is (mean_time, idx), where idx is the index in the corresponding list
    lr_true = np.asarray(match_lists(s2_lr[:,0], s2_true[:,0], dt))
    lr_tp = sum( ~pd.isnull(lr_true[:,0]) & ~pd.isnull(lr_true[:,1]) )
    lr_fp = sum( ~pd.isnull(lr_true[:,0]) & pd.isnull(lr_true[:,1]) )
    lr_fn = sum( pd.isnull(lr_true[:,0]) & ~pd.isnull(lr_true[:,1]) )
    print('Matched LR vs True:  TP,    FP,    FN')
    print('                    {0},   {1},   {2}'.format(lr_tp, lr_fp, lr_fn))

    # Compute the RMSE and plot histogram
    # error list of TP: [s1_dt, true_dt, s1_speed, true_speed, error]
    lr_err = []
    for it1, it2 in lr_true:
        if it1 is not None and it2 is not None:
            lr_err.append([ s2_lr[it1[1],0], s2_true[it2[1],0], s2_lr[it1[1],1], s2_true[it2[1],1],
                            s2_lr[it1[1],1]-s2_true[it2[1],1] ])
    lr_err = np.array(lr_err)
    lr_rmse = np.sqrt( np.sum(lr_err[:,4]**2)/len(lr_err) )

    # -----------------------------------------------------------
    # print out the results and plot
    print('\nadaptLR RMSE: {0:.3f}'.format(lr_rmse))
    plot_hist([lr_err[:,4]], labels=None, title='AdaptLR speed estimation error vs manual labels', xlabel='Speed (mph)')


    plt.show()

def plot_hist( arrs, labels, title='', xlabel='', fontsizes = (22, 18, 16), xlim=None, ylim=None, text_loc=None):
        """
        This function plots the histogram of the list of arrays
        :param arrs: list of arrs
        :param labels: label for each array
        :param title: title
        :param xlabel: xlabel
        :param fontsizes: (title, label, tick)
        :return:
        """
        plt.figure(figsize=(11.5,10))

        colors = itertools.cycle( ['b', 'g', 'm', 'c', 'purple', 'r'])
        text = []
        y_max = 0.0
        std_max = 0.0
        for i, d in enumerate(arrs):
            c = next(colors)
            mu, std = np.mean(d), np.std(d)
            if labels is not None:
                n, bins, patches = plt.hist(d, 50, normed=1, facecolor=c, alpha=0.75, label=labels[i])
            else:
                n, bins, patches = plt.hist(d, 50, normed=1, facecolor=c, alpha=0.75)
            plt.plot(bins, mlab.normpdf(bins, mu, std), color=c, linestyle='--',
                     linewidth=2)
            if labels is not None:
                text.append('{0} mean: {1:.2f}, std: {2:.2f}'.format(labels[i], mu, std))
            else:
                text.append('Bias: {0:.2f}\nStd : {1:.2f}'.format(mu, std))
            # text.append('Mean: {0:.2f}\nStandard deviation: {1:.2f}'.format(mu, std))

            y_max = np.max([y_max, np.max(n)])
            std_max = np.max([std_max, std])

        text_str = '\n'.join(text)
        if text_loc is None:
            plt.text(mu-4*std_max, y_max*1.0, text_str, fontsize=fontsizes[1])
        else:
            plt.text(text_loc[0], text_loc[1], text_str, fontsize=fontsizes[1])

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is None:
            plt.ylim(0, y_max*1.2)
        else:
            plt.ylim(ylim)

        if labels is not None: plt.legend()
        plt.xlabel(xlabel, fontsize=fontsizes[1])
        plt.ylabel('Distribution', fontsize=fontsizes[1])
        plt.title(title, fontsize=fontsizes[0])
        plt.tick_params(axis='both', which='major', labelsize=fontsizes[2])
        plt.draw()


def match_lists(list1, list2, dt):
        """
        This function matches items in two lists: two items are matched if they are separated by less than dt.
        :param list1: a list of datetime
        :param list2: a list of datetime
        :param dt: float, seconds
        :return: matches: a list of tuples; first from list1, second from list2
            (item1, item2), where item1 is [dt, idx], where idx is the idx in list1
            (item1, None), no match in list2
            (None, item2), no match in list1
        """
        l = []
        for i, t in enumerate(list1):
            l.append([t, i, 0])
        for i, t in enumerate(list2):
            l.append([t, i, 1])

        l.sort(key=lambda x: x[0])

        matches = []
        group = [l[0]]
        for i in xrange( len(l)-1 ):
            if abs((l[i][0]-l[i+1][0]).total_seconds()) <= dt and l[i][2] != l[i+1][2]:
                group.append(l[i+1])
            else:
                if len(group) == 1:
                    # One item, did not find match in the other list, find which list it is in
                    if group[0][2] == 0: matches.append( (group[0][0:2], None) )
                    else: matches.append( (None, group[0][0:2]) )
                elif len(group) == 2:
                    # two items, one from each list
                    if group[0][2] == 0: matches.append( (group[0][0:2], group[1][0:2]) )
                    else: matches.append( (group[1][0:2], group[0][0:2]) )
                elif len(group) == 3:
                    # three items, 010 or 101, find which is closer to the center one
                    if abs((group[1][0]-group[0][0]).total_seconds()) <= abs((group[1][0]-group[2][0]).total_seconds()):
                        # group to 01,0 or 10,1
                        if group[1][2] == 0:
                            matches.append( (group[1][0:2], group[0][0:2]) )
                            matches.append( (None, group[2][0:2]) )
                        else:
                            matches.append( (group[0][0:2], group[1][0:2]) )
                            matches.append( (group[2][0:2], None) )
                    else:
                        # group to 0,10 or 1,01
                        if group[1][2] == 0:
                            matches.append( (group[1][0:2], group[2][0:2]) )
                            matches.append( (None, group[0][0:2]) )
                        else:
                            matches.append( (group[2][0:2], group[1][0:2]) )
                            matches.append( (group[0][0:2], None) )
                elif len(group) == 4:
                    # 1010 or 0101, break into 10,10 or 01,01
                    if group[0][2]==0:
                        matches.append( (group[0][0:2], group[1][0:2]) )
                        matches.append( (group[2][0:2], group[3][0:2]) )
                    else:
                        matches.append( (group[1][0:2], group[0][0:2]) )
                        matches.append( (group[3][0:2], group[2][0:2]) )
                elif len(group) == 5:
                    # 01010, or 10101, group to 01,01,0 or 10,10,1
                    # This may not be the optimal solution
                    if group[0][2]==0:
                        matches.append( (group[0][0:2], group[1][0:2]) )
                        matches.append( (group[2][0:2], group[3][0:2]) )
                        matches.append( (group[4][0:2], None) )
                    else:
                        matches.append( (group[1][0:2], group[0][0:2]) )
                        matches.append( (group[3][0:2], group[2][0:2]) )
                        matches.append( (None, group[4][0:2]) )
                elif len(group) == 6:
                    # 101010 or 010101, group into 10,10,10 or 01,01,01
                    if group[0][2]==0:
                        matches.append( (group[0][0:2], group[1][0:2]) )
                        matches.append( (group[2][0:2], group[3][0:2]) )
                        matches.append( (group[4][0:2], group[5][0:2]) )
                    else:
                        matches.append( (group[1][0:2], group[0][0:2]) )
                        matches.append( (group[3][0:2], group[2][0:2]) )
                        matches.append( (group[5][0:2], group[4][0:2]) )
                else:
                    print('Warning: {0} items are grouped together: {1}'.format(len(group), np.array(group)[:,2]))

                group = [l[i+1]]

        # The last group
        if len(group) == 1:
            # One item, did not find match in the other list, find which list it is in
            if group[0][2] == 0: matches.append( (group[0][0:2], None) )
            else: matches.append( (None, group[0][0:2]) )
        elif len(group) == 2:
            # two items, one from each list
            if group[0][2] == 0: matches.append( (group[0][0:2], group[1][0:2]) )
            else: matches.append( (group[1][0:2], group[0][0:2]) )
        elif len(group) == 3:
            # three items, 010 or 101, find which is closer to the center one
            if abs((group[1][0]-group[0][0]).total_seconds()) <= abs((group[1][0]-group[2][0]).total_seconds()):
                # group to 01,0 or 10,1
                if group[1][2] == 0:
                    matches.append( (group[1][0:2], group[0][0:2]) )
                    matches.append( (None, group[2][0:2]) )
                else:
                    matches.append( (group[0][0:2], group[1][0:2]) )
                    matches.append( (group[2][0:2], None) )
            else:
                # group to 0,10 or 1,01
                if group[1][2] == 0:
                    matches.append( (group[1][0:2], group[2][0:2]) )
                    matches.append( (None, group[0][0:2]) )
                else:
                    matches.append( (group[2][0:2], group[1][0:2]) )
                    matches.append( (group[0][0:2], None) )
        elif len(group) == 4:
            # 1010 or 0101
            if group[0][2]==0:
                matches.append( (group[0][0:2], group[1][0:2]) )
                matches.append( (group[2][0:2], group[3][0:2]) )
            else:
                matches.append( (group[1][0:2], group[0][0:2]) )
                matches.append( (group[3][0:2], group[2][0:2]) )
        elif len(group) == 5:
            # This may not be the optimal solution
            # 01010, or 10101, group to 01,01,0 or 10,10,1
            if group[0][2]==0:
                matches.append( (group[0][0:2], group[1][0:2]) )
                matches.append( (group[2][0:2], group[3][0:2]) )
                matches.append( (group[4][0:2], None) )
            else:
                matches.append( (group[1][0:2], group[0][0:2]) )
                matches.append( (group[3][0:2], group[2][0:2]) )
                matches.append( (None, group[4][0:2]) )
        elif len(group) == 6:
            # 101010 or 010101, group into 10,10,10 or 01,01,01
            if group[0][2]==0:
                matches.append( (group[0][0:2], group[1][0:2]) )
                matches.append( (group[2][0:2], group[3][0:2]) )
                matches.append( (group[4][0:2], group[5][0:2]) )
            else:
                matches.append( (group[1][0:2], group[0][0:2]) )
                matches.append( (group[3][0:2], group[2][0:2]) )
                matches.append( (group[5][0:2], group[4][0:2]) )
        else:
            print('Warning: {0} items are grouped together'.format(len(group)))


        return matches




if __name__ == '__main__':
    main()



