"""////////////////////////////////////////////////////////////////////////////
Author: Fangyu Wu (fwu10@illinois.edu)
Date: December 11th, 2015

The TrafficModelsClass currently contains implementation of vehicle detection 
and speed estimation algorithm. Its functionality should be definitely extended
in the future to support vehicle classification, direction detection, and lane
detection.
////////////////////////////////////////////////////////////////////////////"""


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

class TrafficModelsClass:
    def __init__(self, dataset=None):
        PIR, Uson, IMU, Labels = parse(dataset)
        self.PIR = PIR
        self.Uson = Uson
        self.IMU = IMU
        self.Labels = Labels

    # Thresholding ultrasonic signal for vehicle detection
    # For example, to threshold Uson with threshold of 9 on the raw data:
    #
    #    vehicle_detection(Uson, threshold=9, derivative=False)
    #
    # It will return all the time intervals that is believed to contain
    # vehicles. Most of the windows will contain a single vehicle, although
    # some will contain two or three when traffic is in a state of congestion.
    # Note that thresholding derivative is less stable than thresholding the 
    # original Uson signal. Therefore, it is recommended to use the 
    # non-derivative method.
    def vehicle_detection(self, Uson, threshold=2, derivative=True):
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



    # Background subtraction on PIR_data by the median of each pixel
    # Note that dividing by standard deviation is not implemented. This is
    # because that it will decrease the contrast between object pixels and
    # ambient pixels and it is therefore not desirable.
    # 
    # This routine will return the normalized PIR data.
    def background_subtraction(self, PIR_data):
        median = []
        for i in range(len(PIR_data[0])):
            median.append(np.median([t[i] for t in PIR_data]))
        median = np.array(median)
        PIR = []
        for data in PIR_data:
            PIR.append(np.array(data)-median)
        return PIR



    # Estimating slope based on the detection results from ultrasonic signal.
    # It is currently working on test windows that contain only one vehicle
    # with relatively clearly defined thermal path. The fundamental algorithm
    # involves: 
    #   (1) mapping pixels to space, 
    #   (2) reducing data dimension by resampling space pixels based on 
    #       temperature, 
    #   (3) thresholding space pixels by a centain percentile, 
    #   (4) fitting the remaining data with RANSAC regression,
    #   (5) analyzing sensitivity based on the corresponding ultrasonic reading
    #
    # For example, to estimate the slope vehicle from the 101th frame to the
    # 120th frame and display the estimated slope:
    #
    #   slope_estimation(PIR, 100, 120, 3, display=True)
    #
    # It will return the estimated slope within the passing window [100, 120].
    # Currently, PEARL, another multi-model regression is under construction in
    # the hope of replacing RANSAC.
    def slope_estimation(self, PIR, enter, exit, distance, display=False):
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
        
        # Map sensor pixels to space. Resample data based on temperature to
        # convert the temperature dimension to density.
        Time = []
        Pixel = []
        t = 0
        threshold = np.percentile(passing_window, 90)
        for sample in passing_window:
            for idx in np.arange(len(sample)):
                if sample[idx] > threshold:
                    for iter in range(int(sample[idx])):
                        if idx/4 > 0:
                            Time.append(t + np.random.random())
                            Pixel.append(distance_book[idx/4] + 
                                (distance_book[idx/4]-distance_book[idx/4-1]) *
                                np.random.random())
                        elif idx/4 < 0:
                            Time.append(t + np.random.random())
                            Pixel.append(distance_book[idx/4] + 
                                (distance_book[idx/4+1]-distance_book[idx/4]) *
                                np.random.random())
            t += 1

        # Casting to numpy array is required to use the scikit RANSAC model
        Time = np.array(Time).reshape((len(Time),1))
        Pixel = np.array(Pixel).reshape((len(Pixel),1))
        
        # Robustly fit the linear model with RANSAC algorithm
        ransac_model = linear_model.RANSACRegressor(linear_model.LinearRegression(), 
                                                    min_samples=3, max_trials=1000)
        ransac_model.fit(Time, Pixel)
        inlier_mask = ransac_model.inlier_mask_
        outlier_mask = np.labelical_not(inlier_mask)


        # Use the appropriate speed coefficient for sensitivity analysis
        speed_coef = 23
        estimated_speed = speed_coef*ransac_model.estimator_.coef_[0][0]*distance

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

    # Estimating speed based on the slope and ultrasonic measurement.
    # Theoretically, the speed could be directly calculated by 
    # speed = c*slope*distance
    # where c is a constant that may need to be calibrated on site.
    # To estimate the speed of vehicle within the passing intervals:
    #
    # speed_estimation(PIR, Uson, passing_intervals)
    #
    # It will return speeds in list, corresponding to the passing windows that
    # the user supply.
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



    # A function that plot/save snesors status at every timestamp. This needs
    # to be rewritten as per new testing requirement.
    def sensors_status(PIR, Uson, Labels,  
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
        Labels_window = Labels[t-50:t+50]

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
        for idx, label in zip(np.arange(t-50, t+50), Labels_window):
            if label == 10:
                ax21.scatter(idx, 11, marker='o', color='g')
        ax21.axhline(10, color='k', linestyle='--', linewidth=1.5)
        ax21.axvline(t, color='k', linestyle='--', linewidth=1.5)
        ax21.set_xlim(t-50, t+50)
        ax21.set_ylim(0, 20)
        ax21.set_ylabel("Distance (m)")
        ax21.set_title("Utrasonic Signal", fontsize=12)
        
        ax22 = fig.add_subplot(324)
        ax22.plot(np.arange(t-50, t+49), Uson_der_window)
        for idx, label in zip(np.arange(t-50, t+50), Labels_window):
            if label == 10:
                ax22.scatter(idx, 0, marker='o', color='g')
        ax22.axhline(2, color='k', linestyle='--', linewidth=1.5)
        ax22.axhline(-2, color='k', linestyle='--', linewidth=1.5)
        ax22.axvline(t, color='k', linestyle='--', linewidth=1.5)
        ax22.set_xlim(t-50, t+50)
        ax22.set_ylim(-10, 10)
        ax22.set_ylabel("Distance (m)")
        ax22.set_title("Utrasonic Derivative", fontsize=12)

        ax31 = fig.add_subplot(325)
        for idx, label in zip(np.arange(t-50, t+50), Labels_window):
            if label == 10:
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
        for idx, label in zip(np.arange(t-50, t+50), Labels_window):
            if label == 10:
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

# If the class is called, it will analysis every frame within the dataset for
# visualization and video generation.
if __name__ == "__main__":

    # Parse the data into lists
    TModels = TafficModelsClass(dataset)
    PIR = TModels.PIR
    Uson = TModels.Uson
    IMU = TModels.IMU
    Labels = TModels.Labels

    # Vehicle detection
    passing_intervals_original = vehicle_detection(Uson, threshold=10, derivative=False)
    passing_intervals_derivative = vehicle_detection(Uson, threshold=2, derivative=True)
    print "Reported passing from original threshold:", len(passing_intervals_original)
    print "Reported passing from derivative threshold:", len(passing_intervals_derivative)

    # Speed estimation
    passing_speeds_original = speed_estimation(PIR, Uson, passing_intervals_original)
    passing_speeds_derivative = speed_estimation(PIR, Uson, passing_intervals_derivative)

    # Visualize sensors status and save them to files. Disable save by pass
    # "False" into the "save" option.
    for t in np.arange(50,1+50):
        sensors_status(PIR, Uson, Labels, 
                       passing_intervals_original, passing_intervals_derivative,
                       passing_speeds_original, passing_speeds_derivative, t,
                       save=True)

    print "Done." 

