import sys
sys.path.append('./tools')
from _parser import parse
from colormap import colormap
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
from copy import deepcopy
from scipy import stats
from sklearn import linear_model

# Thresholding ultrasonic signal for vehicle detection
def vehicle_detection(Uson, threshold=2, derivative=True):
    print "Detecting vehicles..."
    if derivative:
        Uson_der = [y-x for x, y in zip(Uson[:-1], Uson[1:])] 
        Uson_thres = []
        for data in Uson_der:
            if data < -threshold:
                Uson_thres.append(1) # Vehicle enters
            elif data > threshold:
                Uson_thres.append(-1) # Vehicle leaves
            else:
                Uson_thres.append(0)

        idx = 0
        enter = 0
        exit = 0
        passing_intervals = []
        while idx < len(Uson_thres):
            if Uson_thres[idx] == 1:
                enter = idx
                if idx+1 < len(Uson_thres):
                    idx += 1
                else:
                    break
                while Uson_thres[idx] == 0:
                    if idx+1 < len(Uson_thres):
                        idx += 1
                    else:
                        break
                if Uson_thres[idx] == 1:
                    #print "Missing exit condition at index:", idx
                    pass
                else:
                    exit = idx
                    passing_intervals.append([enter, exit])
                    idx += 1
            elif Uson_thres[idx] == -1:
                #print "Missing enter condition at index:", idx
                idx += 1
            else:
                idx += 1
    else:
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
        
    return passing_intervals



# Background subtraction by pixel median
def background_subtraction(PIR_data):
    median = []
    for i in range(len(PIR_data[0])):
        median.append(np.median([t[i] for t in PIR_data]))
    median = np.array(median)
    PIR = []
    for data in PIR_data:
        PIR.append(np.array(data)-median)
    return PIR



# Estimating slope based on the detection results from ultrasonic signal
def slope_estimation(PIR, enter, exit, display=False):
    passing_window = np.array(PIR[enter-4:exit+5])
    passing_window = background_subtraction(passing_window)
    column_major = np.array([[[i*64+k*16+j for k in range(4)] for j in range(16)]
                            for i in range(3)]).reshape(192)
    colormap_col = []
    for line in passing_window:
        col = []
        for i in column_major:
            col.append(line[i])
        colormap_col.append(col)
    passing_window = np.array(colormap_col)
    #passing_window = passing_window[:,64:128]
    
    Time = []
    Pixel = []
    t = 0
    # This assumes that vehicles are hotter than ambient temperature
    threshold = np.percentile(passing_window, 90)
    for sample in passing_window:
        for idx in range(len(sample)):
            if sample[idx] > threshold:
                Time.append(t)
                Pixel.append(idx)
        t += 1

    
    # Robustly fit the linear model with RANSAC algorithm
    ransac_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
    Time = np.array(Time).reshape((len(Time),1))
    Pixel = np.array(Pixel).reshape((len(Pixel),1))
    ransac_model.fit(Time, Pixel)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    # Fit data into a linear regression model
    #lr_model = linear_model.LinearRegression()
    #lr_model.fit(Time, Pixel)
    
    # Prediction based on ransac algorithm
    ransac_Pixel = ransac_model.predict(Time)
    # Prediction based on linear regression
    #lr_Pixel = lr_model.predict(Time)

    if display:
        plt.figure()
        plt.imshow(np.transpose(passing_window), interpolation="nearest",
                   aspect="auto", origin="lower", vmin=-2, vmax=8, cmap="gray")
        plt.scatter(Time[inlier_mask], Pixel[inlier_mask], color='g', label="Inliers")
        plt.scatter(Time[outlier_mask], Pixel[outlier_mask], color='r', label="Outliers")
        plt.plot(Time, ransac_Pixel, 'c', label="Robust LR", linewidth=1.5)
        #plt.plot(Time, lr_Pixel, 'b', label="LR", linewidth=1.5)
        plt.xlim(-0.5, passing_window.shape[0]-0.5)
        plt.ylim(-0.5, passing_window.shape[1]-0.5)
        
        plt.xlabel("Time (0.125 sec)")
        plt.ylabel("Pixels")
        plt.title("Linear Regression Results")
        plt.legend(loc="lower right")
        plt.show()
    
    return ransac_model.estimator_.coef_[0][0]



# Estimating speed based on the slope and ultrasonic measurement
def speed_estimation(PIR, Uson, passing_intervals):
    print "Estimating speed..."

    Speeds = []
    for interval in passing_intervals:
        enter = interval[0]
        exit = interval[1]
        s = slope_estimation(PIR, enter, exit, display=False)
        d= np.min(Uson[enter:exit])

        k = 2 # speed coefficient
        Speeds.append(k*s*d)

    # Leave off negative estimates
    average = np.mean(Speeds)
    for idx in range(len(Speeds)):
        if Speeds[idx] < 0:
            Speeds[idx] = average
        
    Speeds = [[Speeds[idx]]*(passing_intervals[idx][1]-passing_intervals[idx][0]) 
              for idx in range(len(Speeds))]
    return Speeds



# A function that plot/save snesors status at every timestamp
def sensors_status(PIR, Uson, Log,  
                   intervals_original, intervals_derivative,
                   speeds_original, speeds_derivative, t, save=False):
    if t < 50:
        raise ValueError("Time %d is less than 50." % t)
    if t > len(PIR)-51:
        raise ValueError("Time %d is larger than %d." % (t, len(PIR)-50))
    

    # Obtain colormap
    PIR_colormap = background_subtraction(np.array(PIR[t-50:t+50]))
    column_major = np.array([[[i*64+k*16+j for k in range(4)] for j in range(16)]
                            for i in range(3)]).reshape(192)
    tmp = []
    for line in PIR_colormap:
        col = []
        for i in column_major:
            col.append(line[i])
        tmp.append(col)
    PIR_colormap = np.transpose(tmp)
    
    # Obtain ultrasonic window and its derivative window
    Uson_window = [data for data in Uson[t-50:t+50]]
    Uson_der_window = [y-x for x, y in zip(Uson_window[:-1], Uson_window[1:])] 
    Log_window = Log[t-50:t+50]

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax11 = fig.add_subplot(321)
    ax11.imshow(PIR_colormap, interpolation='nearest', cmap="winter", 
                vmin=-2, vmax=5, aspect="auto", origin="lower", 
                extent=[t-50, t+50, -0.5, 191.5])
    ax11.axvline(t, color='k', linestyle='--', linewidth=1.5)
    for time in intervals_original:
        if time[0] in np.arange(t-50, t+50):
            ax11.axvline(time[0]-4, color='r')
        if time[1] in np.arange(t-50, t+50):
            ax11.axvline(time[1]+4, color='r')
    ax11.set_ylabel("Pixels")
    ax11.set_title("Regression Window from Uson", fontsize=12)
    
    ax12 = fig.add_subplot(322)
    ax12.imshow(PIR_colormap, interpolation='nearest', cmap="winter", 
                vmin=-2, vmax=5, aspect="auto", origin="lower",  
                extent=[t-50, t+50, -0.5, 191.5])
    ax12.axvline(t, color='k', linestyle='--', linewidth=1.5)
    for time in intervals_derivative:
        if time[0] in np.arange(t-50, t+50):
            ax12.axvline(time[0]-4, color='r')
        if time[1] in np.arange(t-50, t+50):
            ax12.axvline(time[1]+4, color='r')
    ax12.set_ylabel("Pixels")
    ax12.set_title("Regression Window from Uson Derivative", fontsize=12)
    
    ax21 = fig.add_subplot(323)
    ax21.plot(np.arange(t-50, t+50), Uson_window)
    for idx, log in zip(np.arange(t-50, t+50), Log_window):
        if log == 10:
            ax21.scatter(idx, 11, marker='o', color='g')
    ax21.axhline(10, color='k', linestyle='--', linewidth=1.5)
    ax21.axvline(t, color='k', linestyle='--', linewidth=1.5)
    ax21.set_xlim(t-50, t+50)
    ax21.set_ylim(0, 20)
    ax21.set_ylabel("Distance (m)")
    ax21.set_title("Utrasonic Signal", fontsize=12)
    
    ax22 = fig.add_subplot(324)
    ax22.plot(np.arange(t-50, t+49), Uson_der_window)
    for idx, log in zip(np.arange(t-50, t+50), Log_window):
        if log == 10:
            ax22.scatter(idx, 0, marker='o', color='g')
    ax22.axhline(2, color='k', linestyle='--', linewidth=1.5)
    ax22.axhline(-2, color='k', linestyle='--', linewidth=1.5)
    ax22.axvline(t, color='k', linestyle='--', linewidth=1.5)
    ax22.set_xlim(t-50, t+50)
    ax22.set_ylim(-10, 10)
    ax22.set_ylabel("Distance (m)")
    ax22.set_title("Utrasonic Derivative", fontsize=12)

    ax31 = fig.add_subplot(325)
    for idx, log in zip(np.arange(t-50, t+50), Log_window):
        if log == 10:
            ax31.scatter(idx, 35, marker='o', color='g')
    for time, speed in zip(intervals_original, speeds_original):
        for moment, mm_speed in zip(np.arange(time[0], time[1]), speed):
            if moment in np.arange(t-50, t+50):
                ax31.scatter(moment, mm_speed, marker='o', color='b')
                break
    ax31.axhline(35, color='k', linestyle='--', linewidth=1.5)
    ax31.axvline(t, color='k', linestyle='--', linewidth=1.5)
    ax31.set_xlim(t-50, t+50)
    ax31.set_ylim(0, 150)
    ax31.set_xlabel("Time (0.125 sec)")
    ax31.set_ylabel("Speed (mph)")
    ax31.set_title("Speeds from Uson", fontsize=12)

    ax32 = fig.add_subplot(326)
    for idx, log in zip(np.arange(t-50, t+50), Log_window):
        if log == 10:
            ax32.scatter(idx, 35, marker='o', color='g')
    for time, speed in zip(intervals_derivative, speeds_derivative):
        for moment, mm_speed in zip(np.arange(time[0], time[1]), speed):
            if moment in np.arange(t-50, t+50):
                ax32.scatter(moment, mm_speed, marker='o', color='b')
                break
    ax32.axhline(35, color='k', linestyle='--', linewidth=1.5)
    ax32.axvline(t, color='k', linestyle='--', linewidth=1.5)
    ax32.set_xlim(t-50, t+50)
    ax32.set_ylim(0, 150)
    ax32.set_xlabel("Time (0.125 sec)")
    ax32.set_ylabel("Speed (mph)")
    ax32.set_title("Speeds from Uson Derivative", fontsize=12)
    
    if save:
        plt.savefig("visualization/sensor_status/%06d.png" % t)
        plt.close(fig)
    else:
        plt.show()



# Parse the data into lists
PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
PIR = deepcopy(PIR_data)
Uson = [data.uson for data in IMUU_reduced_data[0:]]
Log = deepcopy(LOG_inflated_data)

# Vehicle detection
passing_intervals_original = vehicle_detection(Uson, threshold=10, derivative=False)
passing_intervals_derivative = vehicle_detection(Uson, threshold=2, derivative=True)
print "Reported passing from original threshold:", len(passing_intervals_original)
print "Reported passing from derivative threshold:", len(passing_intervals_derivative)

# Speed estimation
passing_speeds_original = speed_estimation(PIR, Uson, passing_intervals_original)
passing_speeds_derivative = speed_estimation(PIR, Uson, passing_intervals_derivative)

# Report sensors status

for t in np.arange(50,4875+50):
    sensors_status(PIR, Uson, Log, 
                   passing_intervals_original, passing_intervals_derivative,
                   passing_speeds_original, passing_speeds_derivative, t, save=True)

print "Done." 
