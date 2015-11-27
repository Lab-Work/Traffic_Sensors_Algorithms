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

# Nonlinear trasnformation factor
# Left PIR center
center_left = np.pi/180 * (-42)
# Middle PIR center
center_mid = np.pi/180 * 0
# Right PIR center
center_right = np.pi/180 * 42

transform = np.arange(-7.5, 8.5) * np.pi/180 * 60 / 16
transform_left = [np.arctan(angle) for angle in transform+center_left]
transform_mid = [np.arctan(angle) for angle in transform+center_mid]
transform_right = [np.arctan(angle) for angle in transform+center_right]

#transform_base = transform_mid[8]
#transform_left = [factor/transform_base for factor in transform_left]
#transform_mid = [factor/transform_base for factor in transform_mid]
#transform_right = [factor/transform_base for factor in transform_right]

plt.figure()
plt.scatter(np.arange(16), transform_left)
plt.scatter(np.arange(16, 32), transform_mid)
plt.scatter(np.arange(32, 48), transform_right)
plt.show()

distance_book = np.concatenate((transform_left, transform_mid, transform_right))

def find_slope(PIR_data, begin, end, display=False):
    #print "Finding PIR Slope..."
    passing_window = np.array(PIR_data[begin-8:end+8])
    
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
    threshold = np.percentile(passing_window, 90)
    for sample in passing_window:
        for idx in range(len(sample)):
            if sample[idx] > threshold:
                Time.append(t)
                Pixel.append(distance_book[idx/4])
        t += 1
    
    # Transform data into linear space
    Time = np.array(Time).reshape((len(Time),1))
    Pixel = np.array(Pixel).reshape((len(Pixel),1))
    
    
    
    # Robustly fit the linear model with RANSAC algorithm
    ransac_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
    ransac_model.fit(Time, Pixel)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    # Prediction based on ransac algorithm
    ransac_Pixel = ransac_model.predict(Time)

    if display:
        plt.figure()
        plt.scatter(Time[inlier_mask], Pixel[inlier_mask], color='g', label="Inliers")
        plt.scatter(Time[outlier_mask], Pixel[outlier_mask], color='r', label="Outliers")
        plt.plot(Time, ransac_Pixel, 'c', label="Robust LR", linewidth=1.5)
        #plt.xlim(-0.5, passing_window.shape[0]-0.5)
        #plt.ylim(-0.5, passing_window.shape[1]-0.5)
        plt.xlabel("Time (0.125 sec)")
        plt.ylabel("Pixels")
        plt.title("Linear Regression Results")
        plt.legend(loc="lower right")
        plt.show()
    
    return ransac_model.estimator_.coef_[0][0]

PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
IMUU_reduced_data = [data.uson for data in IMUU_reduced_data]

MEAN = []
for i in range(len(PIR_data[0])):
    MEAN.append(np.mean([t[i] for t in PIR_data]))
MEAN = np.array(MEAN)
background_subtracted_PIR = []
for data in PIR_data:
    background_subtracted_PIR.append(np.array(data)-MEAN)
background_subtracted_PIR = np.array(background_subtracted_PIR)

threshold = 2
Uson = deepcopy(IMUU_reduced_data)
Uson_grad = [y-x for x, y in zip(Uson[:-1], Uson[1:])] 
Veh = []
for x in Uson_grad:
    if x < -threshold:
        Veh.append(1) # Vehicle enters
    elif x > threshold:
        Veh.append(-1) # Vehicle leaves
    else:
        Veh.append(0)

idx = 0
count = []
enter = 0
exit = 0
while idx < len(Veh):
    if Veh[idx] == 1:
        enter = idx
        if idx+1 < len(Veh):
            idx += 1
        else:
            break
        while Veh[idx] == 0:
            if idx+1 < len(Veh):
                idx += 1
            else:
                break
        if Veh[idx] == 1:
            pass
        else:
            exit = idx
            count.append([enter, exit])
            idx += 1
    elif Veh[idx] == -1:
        idx += 1
    else:
        idx += 1

cumulat = [0]*len(Veh)
for x in count:
    cumulat[x[1]] = 1
for i in range(len(cumulat)-1):
    cumulat[i+1] += cumulat[i]

if True:
    S = []
    D = []
    for x in count:
        s = find_slope(background_subtracted_PIR, x[0], x[1], False)
        d= np.min(Uson[x[0]:x[1]+1])
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
