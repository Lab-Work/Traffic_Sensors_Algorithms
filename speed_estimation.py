import sys
sys.path.append('./tools')
from _parser import parse
from colormap import colormap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
import numpy as np
from copy import deepcopy
import cookb_signalsmooth as ss
from scipy import stats
from sklearn import linear_model


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
    smoothed_passing_window = passing_window
    
    T = []
    P = []
    time = 0
    threshold = np.percentile(smoothed_passing_window, 90)
    for sample in smoothed_passing_window:
        for idx in range(len(sample)):
            if sample[idx] > threshold:
                T.append(time)
                P.append(idx)
        time += 1

    # Fit data into a linear regression model
    lr_model = linear_model.LinearRegression()
    T = np.array(T).reshape((len(T),1))
    P = np.array(P).reshape((len(P),1))
    lr_model.fit(T, P)

    # Robustly fit the linear model with RANSAC algorithm
    ransac_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
    ransac_model.fit(T, P)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    # Prediction based on linear regression
    lr_P = lr_model.predict(T)
    # Prediction based on ransac algorithm
    ransac_P = ransac_model.predict(T)

    if display:
        plt.figure()
        plt.imshow(np.transpose(smoothed_passing_window), interpolation="nearest",
                   aspect="auto", origin="lower", cmap="gray", vmin=-2, vmax=5)
        plt.scatter(T[inlier_mask], P[inlier_mask], color='g', label="Inliers")
        plt.scatter(T[outlier_mask], P[outlier_mask], color='r', label="Outliers")
        #plt.plot(T, lr_P, 'b', label="Linear Regression", linewidth=2.0)
        plt.plot(T, ransac_P, 'r', label="Robust LR", linewidth=2.0)
        plt.xlim(-0.5, smoothed_passing_window.shape[0]-0.5)
        plt.ylim(-0.5, smoothed_passing_window.shape[1]-0.5)
        
        plt.xlabel("Time (0.125 sec)")
        plt.ylabel("Pixels")
        plt.title("Linear Regression Results")
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, ncol=4)
        plt.show()
    #print lr_model.coef_
    #print ransac_model.estimator_.coef_
    return [lr_model.coef_[0], ransac_model.estimator_.coef_]


PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
IMUU_reduced_data = [data.uson for data in IMUU_reduced_data]
#colormap(PIR_data, IMUU_reduced_data, LOG_inflated_data, -1,
#         save_fig=False, smooth=True)

MEAN = []
for i in range(len(PIR_data[0])):
    MEAN.append(np.mean([t[i] for t in PIR_data]))
MEAN = np.array(MEAN)
STDEV = []
for i in range(len(PIR_data[0])):
    STDEV.append(np.std([t[i] for t in PIR_data]))
STDEV = np.array(STDEV)
background_subtracted_PIR = []
for data in PIR_data:
    background_subtracted_PIR.append((np.array(data)-MEAN)/STDEV)
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
            print "Missing exit condition at index:", idx
        else:
            exit = idx
            count.append([enter, exit])
            idx += 1
    elif Veh[idx] == -1:
        print "Missing enter condition at index:", idx
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
        s = find_slope(background_subtracted_PIR, x[0], x[1], True)
        #print "Slope:", s
        d= np.min(Uson[x[0]:x[1]+1])
        S.append(s)
        D.append(d)

    S = np.array(S)
    lr_mean = np.mean(S[:,0])
    ransac_mean = np.mean(S[:,1])
    lr_std = np.std(S[:,0])
    ransac_std = np.std(S[:,1])
    print "LR measures slope of %.2f +/- %.2f 8px/sec" % (lr_mean, lr_std)
    print "Robust LR measures slope of %.2f +/- %.2f 8px/sec" % (ransac_mean, ransac_std)
    
    # Leave off unrealistic estimates based on common sense
    for est_idx in range(len(S[:,0])):
        if S[est_idx][0] < 1:
            S[est_idx][0] = lr_mean 
    for est_idx in range(len(S[:,1])):
        if S[est_idx][1] < 1:
            S[est_idx][1] = lr_mean

    plt.figure()
    plt.plot(S[:,0], 'b', label="LR Slope", linewidth=2.0)
    plt.plot(S[:,1], 'r', label="Robust LR Slope", linewidth=2.0)
    #plt.plot(D, label="Normal distance")
    plt.axhline(lr_mean, linestyle='--', color='b', label="LR Average")
    plt.axhline(ransac_mean, linestyle='--', color='r', 
                label="Robust LR Average")
    plt.xlabel("Nth Vehicle Passing")
    plt.ylabel("Slope (8px/sec)")
    plt.title("Slope Estimation")
    plt.legend()
    plt.show()
