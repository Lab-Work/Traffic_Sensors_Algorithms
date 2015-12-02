import sys
sys.path.append('./tools')
from _parser import parse
from colormap import colormap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
import numpy as np
from copy import deepcopy
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from skimage.filters import roberts, sobel


# Nonlinear trasnformation factor
# Left PIR center
center_left = np.pi/180 * (-42)
# Middle PIR center
center_mid = np.pi/180 * 0
# Right PIR center
center_right = np.pi/180 * 42

transform = np.arange(-7.5, 8.5) * np.pi/180 * 60 / 16
transform_left = [np.tan(angle) for angle in transform+center_left]
transform_mid = [np.tan(angle) for angle in transform+center_mid]
transform_right = [np.tan(angle) for angle in transform+center_right]
distance_book = np.concatenate((transform_left, transform_mid, transform_right))

# Background subtraction by pixel median
def background_subtraction(PIR_data):
    #print "Running background subtraction"
    #plt.figure()
    #plt.imshow(PIR_data)
    median = []
    for i in range(len(PIR_data[0])):
        median.append(np.median([t[i] for t in PIR_data]))
    median = np.array(median)
    #stdev = []
    #for i in range(len(PIR_data[0])):
    #    stdev.append(np.std([t[i] for t in PIR_data]))
    #stdev = np.array(stdev)
    PIR = []
    for data in PIR_data:
        PIR.append(np.array(data)-median)
    #plt.figure()
    #plt.imshow(_PIR)
    #plt.show()
    return PIR


def find_slope(PIR_data, begin, end, distance, display=False):
    #print "Finding colormap slope..."
    passing_window = np.array(PIR_data[begin-8:end+8])
    passing_window = background_subtraction(passing_window)

    # Change PIR data from row major to column major
    column_major = np.array([[[i*64+k*16+j for k in range(4)] for j in range(16)]
                            for i in range(3)]).reshape(192)
    colormap_col = []
    for line in passing_window:
        col = []
        for i in column_major:
            col.append(line[i])
        colormap_col.append(col)
    passing_window = np.array(colormap_col)
    
    Time = []
    Pixel = []
    t = 0
    # This assumes that vehicles are hotter than ambient temperature
    #threshold = np.percentile(passing_window, 85)
    #for sample, left_sample, right_sample in zip(passing_window[1:-1], 
    #                                             passing_window[:-1],
    #                                             passing_window[1:]):
    #    for idx in np.arange(3, len(sample)-3):
    #        if (np.sum(sample[idx-3:idx+3]) + 
    #            np.sum(left_sample[idx-1:idx+1]) + 
    #            np.sum(right_sample[idx-1:idx+1])) > 6*threshold:
    #            Time.append(t)
    #            Pixel.append(distance_book[idx/4])
    #            #Pixel.append(idx)
    #    t += 1
    
    threshold = np.percentile(passing_window, 80)
    for sample in passing_window:
        for idx in np.arange(len(sample)):
            if sample[idx] > threshold:
                for iter in range(int(sample[idx])):
                    Time.append(t + np.random.random())
                    Pixel.append(distance_book[idx/4] * 
                                 (1 + 0.05*np.random.random()))
        t += 1

    Time = np.array(Time).reshape((len(Time),1))
    Pixel = np.array(Pixel).reshape((len(Pixel),1))
    
    # Robustly fit the linear model with RANSAC algorithm
    ransac_model = linear_model.RANSACRegressor(linear_model.LinearRegression(), 
                                                min_samples=3, max_trials=1000)
    ransac_model.fit(Time, Pixel)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    speed_coef = 23
    estimated_speed = speed_coef*ransac_model.estimator_.coef_[0][0]*distance

    # Prediction based on ransac algorithm
    #ransac_Pixel = ransac_model.predict(Time)

    # Sensitivity analysis
    slope = ransac_model.estimator_.coef_[0][0]
    intercept = ransac_model.estimator_.intercept_[0]
    ransac_Pixel = Time*slope + intercept
    # 5 mph higher than the estimate
    slope_higher = (estimated_speed + 5) / (speed_coef*distance)
    ransac_Pixel_higher = Time*slope_higher + intercept
    shift_higher = np.median(ransac_Pixel) - np.median(ransac_Pixel_higher)
    ransac_Pixel_higher += shift_higher
    # 5 mph lower than the estimate
    slope_lower = (estimated_speed - 5) / (speed_coef*distance)
    ransac_Pixel_lower = Time*slope_lower + intercept
    shift_lower = np.median(ransac_Pixel_lower) - np.median(ransac_Pixel)
    ransac_Pixel_lower -= shift_lower

    if False:
        # Experiment with edge detection
        edge_roberts = roberts(passing_window)
        edge_sobel = sobel(passing_window)

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.imshow(edge_roberts, cmap="gray")
        ax1.set_title("Roberts")
        ax2 = fig.add_subplot(312)
        ax2.imshow(edge_sobel, cmap="gray")
        ax2.set_title("Sobel")
        ax3 = fig.add_subplot(313)
        ax3.imshow(passing_window, cmap="gray")
        ax3.set_title("Original")
        plt.show()

    if display:
        topo_Y = np.concatenate((distance_book, distance_book))
        topo_Y = np.concatenate((topo_Y, topo_Y))
        topo_Y = np.sort(topo_Y)
        topo_X = np.arange(passing_window.shape[0])
        topo_X, topo_Y = np.meshgrid(topo_X, topo_Y)
        topo_Z = np.transpose(passing_window)

        fig = plt.figure(figsize=(15, 5), dpi=100)
        ax0 = fig.add_subplot(121, projection="3d")
        ax0.contourf(topo_X, topo_Y, topo_Z, 100, vmin=-5, vmax=20)
        ax0.plot(Time, ransac_Pixel, 22, 'r')
        #ax0.plot_surface(topo_X, topo_Y, topo_Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        #                 linewidth=0, antialiased=False, vmin=-5, vmax=20)
        ax0.set_xlabel("Time (0.125 sec)")
        ax0.set_ylabel("Norminal Distance from Centerline")
        ax0.set_zlabel("Nominal Temperature")
        ax0.set_title("Measured Distance = %.2f m" % distance)

        ax = fig.add_subplot(122)
        ax.set_xlim(0, passing_window.shape[0]-1)
        ax.set_ylim(-np.tan(2*np.pi/5)+0.3, np.tan(2*np.pi/5)-0.3)
        ax.set_xlabel("Time (0.125 sec)")
        ax.set_ylabel("Norminal Distance from Centerline")
        ax.set_title("Estimated Slope = %.2f /sec Estimated Speed = %.2f mph" 
                     % (ransac_model.estimator_.coef_[0][0], estimated_speed))
        
        ax.scatter(Time[inlier_mask], Pixel[inlier_mask], 
                   marker='.', color='g', label="Inliers")
        ax.scatter(Time[outlier_mask], Pixel[outlier_mask], 
                   marker='.', color='r', label="Outliers")
        ax.plot(Time, ransac_Pixel, 'c', label="Robust LR", linewidth=1.5)
        ax.plot(Time, ransac_Pixel_higher, '--r', label="Robust LR High", linewidth=1.5)
        ax.plot(Time, ransac_Pixel_lower, '--b', label="Robust LR Low", linewidth=1.5)
        #ax.legend(loc="lower right")
        
        plt.show()
    
    return ransac_model.estimator_.coef_[0][0]

PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
IMUU_reduced_data = [data.uson for data in IMUU_reduced_data]

threshold = 9
Uson = deepcopy(IMUU_reduced_data)
idx = 0
enter = 0
exit = 0
passing_intervals = []
while idx < len(Uson):
    if Uson[idx] < threshold:
        enter = idx
        if idx+1 < len(Uson):
            idx += 1
        else:
            break
        while Uson[idx] < threshold:
            if idx+1 < len(Uson):
                idx += 1
            else:
                break
        exit = idx
        passing_intervals.append([enter, exit])
        idx += 1
    else:
        idx += 1
S = []
D = []
for x in passing_intervals:
    d= np.min(Uson[x[0]:x[1]+1])
    s = find_slope(PIR_data, x[0], x[1], d, True)
    S.append(s)
    D.append(d)

S = np.array(S)
D = np.array(D)
V = S*D 
ransac_mean = np.mean(S)
ransac_std = np.std(S)
print "Robust LR measures slope of %.2f +/- %.2f 8px/sec" % (ransac_mean, ransac_std)

plt.figure()
plt.plot(S, 'r', label="Robust LR Slope", linewidth=2.0)
#plt.plot(D, label="Normal distance")
plt.plot(V, label="Speed estimation")
plt.axhline(ransac_mean, linestyle='--', color='r', 
            label="Robust LR Average")
plt.xlabel("Nth Vehicle Passing")
plt.ylabel("Slope (8px/sec)")
plt.title("Estimation")
plt.legend()
plt.show()
