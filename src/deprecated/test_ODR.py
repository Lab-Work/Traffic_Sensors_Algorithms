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

def f(B, x):
    # Linear function y = m*x + b
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

# set the line
# true line: y = a*x + b
a = -5.0
b = 2.5
len_x = 33
num_samples = 500

x_true = np.linspace(-0.26,0.26,len_x)
y_true = a*x_true + b
y_mean = []

x = []
y = []
# generate noisy data points
for v in x_true:

    if -0.1 < v < 0.1:
        # fewer data points
        n_s = 10
        x = x + [v]*n_s
        random_y = a*v+b + np.random.normal(0,0.5,n_s)
    else:
        x = x + [v]*num_samples
        random_y = a*v+b + np.random.normal(0,0.5,num_samples)

    y_mean.append(np.mean(random_y))
    y = y + random_y.tolist()

x = np.asarray(x)
y = np.asarray(y)
y_mean = np.asarray(y_mean)

print len(x)
print len(y)

# ==============================================
if False:
    # ODR regression
    linear = odr.Model(f)
    mydata = odr.Data(x, y)
    myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
    myoutput = myodr.run()
    myoutput.pprint()
    _slope, _intercept = myoutput.beta
    # print('y = {0} x + {1}'.format(_slope, _intercept))
    _r_value = myoutput.res_var
    r2 = 1-_r_value

    y_fit = _slope*x + _intercept

    print('ODR result: y = {0} x + {1}'.format(_slope, _intercept))

# ===============================================
# Linear regression
_slope, _intercept, _r_value, _p_value, _std_err = scipy.stats.linregress(x,y)
# express in space = coe*time + intercept
line = np.array([ _slope, _intercept])
r2 = _r_value ** 2

y_linear_fit = _slope*x + _intercept
# y_linear_inv_fit = -(1.0/_slope) * x

print('Linear result: y = {0} x + {1}'.format(_slope, _intercept))
# print('Linear slope: {0}, inverse slope: {1}'.format(_slope, -(1.0/_slope)))

# ===============================================
# test to understand the output of ODR package

# print('ey = y_data - y_fit: {0}'.format(y-y_fit))
ey2 = np.power(y - y_linear_fit, 2)
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

print('Manual calculation:')
print('       residual sum:{0}'.format(sum_ey2))
print('       total sum:{0}'.format(y_res_sum2))
print('       R2: {0}'.format(1-sum_ey2/y_res_sum2))
print('       correlation coe: {0}'.format(cor_coe))
print('       r2: {0}\n'.format(cor_coe**2))
print('       mean residual sum: {0}')

# # print('ODR eps:')
# # print(myoutput.eps)
# print('ODR results:')
# print('      sum_square:{0}'.format(myoutput.sum_square))
# print('      sum_square_delta:{0}'.format(myoutput.sum_square_delta))
# print('      sum_square_eps:{0}'.format(myoutput.sum_square_eps))
# print('      res_var:{0}\n'.format(myoutput.res_var))

print('Linear results:')
print('      Linear r_value: {0}'.format(_r_value))
print('      Linear r2:{0}'.format(r2))

# ===============================================
plt.figure()
plt.scatter(x,y)
# plt.plot(x_true,y_true,color='g',linewidth=2, label='true')
# plt.plot(x, y_fit,color='r',linewidth=2,label='odr')
plt.plot(x, y_linear_fit,color='b',linewidth=2,label='linear')
plt.scatter(x_true, y_mean, color='r')
# plt.plot(x, y_linear_inv_fit,color='g',linewidth=2, label='linear_inv')
plt.legend()
plt.xlim([-0.4, 0.4])
plt.ylim([0,5])
plt.show()



























