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
a = -0.1
b = 2.5
x_true = np.linspace(-0.01,0.01,3)
y_true = a*x_true + b

# for each x value, scatter several response values
x = np.zeros(0)
y = np.zeros(0)
for i in xrange(100):
    x = np.concatenate([x, x_true + np.random.normal(0,0.0,3)])
    y = np.concatenate([y, y_true + np.random.normal(0,2,3)])

# ==============================================
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
y_linear_inv_fit = -(1.0/_slope) * x

print('Linear result: y = {0} x + {1}'.format(_slope, _intercept))
print('Linear slope: {0}, inverse slope: {1}'.format(_slope, -(1.0/_slope)))

# ===============================================
# test to understand the output of ODR package

# print('ey = y_data - y_fit: {0}'.format(y-y_fit))
ey2 = np.power(y - y_linear_fit, 2)
sum_ey2 = np.sum(ey2)

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
print('       correlation coe: {0}\n'.format(cor_coe))

# print('ODR eps:')
# print(myoutput.eps)
print('ODR results:')
print('      sum_square:{0}'.format(myoutput.sum_square))
print('      sum_square_delta:{0}'.format(myoutput.sum_square_delta))
print('      sum_square_eps:{0}'.format(myoutput.sum_square_eps))
print('      res_var:{0}\n'.format(myoutput.res_var))

print('Linear results:')
print('      Linear r_value: {0}'.format(_r_value))
print('      Linear r2:{0}'.format(r2))

# ===============================================
plt.figure()
plt.scatter(x,y)
# plt.plot(x_true,y_true,color='g',linewidth=2, label='true')
plt.plot(x, y_fit,color='r',linewidth=2,label='odr')
plt.plot(x, y_linear_fit,color='b',linewidth=2,label='linear')
plt.plot(x, y_linear_inv_fit,color='g',linewidth=2, label='linear_inv')
plt.legend()
plt.xlim([-0.26, 0.26])
plt.ylim([0,5])
plt.show()



























