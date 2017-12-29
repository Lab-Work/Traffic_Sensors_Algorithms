import cv2
print('Imported OpenCV version {0}'.format(cv2.__version__))
import sys, os
from contextlib import contextmanager
import time, glob
from copy import deepcopy
from os.path import exists
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
from matplotlib import gridspec
import matplotlib.patches as patches
from datetime import datetime
from datetime import timedelta
import numpy as np
import scipy
import itertools
from scipy import stats
from scipy import odr
from scipy.signal import argrelextrema
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
import pandas as pd
import imutils


"""
This script is to test the orthogonal distance regression
"""

# ================================================================
# Define the dataset to regress
# true line: y = a*x + b
a = -5.0
b = 2.5
len_x = 33
x_true = np.linspace(-0.26,0.26,len_x)
y_true = a*x_true + b

# number of samples at each level
n_samples = 3
mu = 0.0
std = 0.5

x = []
y = []
y_mean = []
y_median = []
# generate noisy data points
for v in x_true:

    x = x + [v]*n_samples
    random_y = a*v+b + np.random.normal(mu, std, n_samples)

    y_mean.append(np.mean(random_y))
    y_median.append(np.median(random_y))
    y = y + random_y.tolist()

x = np.asarray(x)
y = np.asarray(y)

# compute teh mean of the data
y_mean = np.asarray(y_mean)


# ================================================================
# Linear regression using all data
_slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x,y)
# express in space = coe*time + intercept
line = np.array([ _slope, _intercept])
r2 = _r_value ** 2

y_lr_fit = _slope*x + _intercept

print('Linear all data: y = {0} x + {1}'.format(_slope, _intercept))

# print('ey = y_data - y_fit: {0}'.format(y-y_fit))
ey2 = np.power(y - y_lr_fit, 2)
sum_ey2 = np.sum(ey2)

# compute the sum squared error of (mean(y) - y_fit) for each x value
mean_ey = np.power( y_mean - (_slope*x_true+_intercept), 2)
sum_mean_ey = np.sum(mean_ey)

y_res = y - np.mean(y)
y_res_sum2 = np.sum(np.power(y_res, 2))

# compute the correlation coefficient
sig_x = np.std(x)
sig_y = np.std(y)
cov_xy = np.mean(x*y) - np.mean(x)*np.mean(y)
cor_coe = cov_xy/(sig_x*sig_y)

# print('LR: Manual calculation:')
# print('       residual sum:{0}'.format(sum_ey2))
# print('       total sum:{0}'.format(y_res_sum2))
# print('       R2: {0}'.format(1-sum_ey2/y_res_sum2))
# print('       correlation coe: {0}'.format(cor_coe))
# print('       r2: {0}\n'.format(cor_coe**2))
# print('       mean residual sum: {0}')

print('LR r2:    {0}'.format(r2))

# ================================================================
# Linear regression using mean point
_slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x_true, y_mean)
line_mean = np.array([ _slope, _intercept])
y_mean_fit = _slope*x_true + _intercept

print('Linear mean data: y = {0} x + {1}'.format(_slope, _intercept))
print('Mean r2:  {0}'.format(_r_value**2))

# ================================================================
# Linear regression using median point
_slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x_true, y_median)
line_median = np.array([ _slope, _intercept])
y_median_fit = _slope*x_true + _intercept

print('Linear median data: y = {0} x + {1}'.format(_slope, _intercept))
print('Median r2:{0}'.format(_r_value**2))


# ================================================================
# visualization
plt.figure()
# all data points
plt.scatter(x,y, color='g')
# mean in red
plt.scatter(x_true, y_mean, color='r')
# median in
plt.scatter(x_true, y_median, color='b')

# plot LR
plt.plot(x, y_lr_fit,linewidth=2,color='g', label='linear', linestyle='-')
plt.plot(x_true, y_mean_fit,linewidth=2,color='r', label='mean', linestyle='--')
plt.plot(x_true, y_median_fit,linewidth=2,color='b', label='median', linestyle='--')

plt.legend()
plt.xlim([-0.4, 0.4])
plt.ylim([0,5])
plt.show()



























