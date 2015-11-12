import sys
sys.path.append('./tools')
from _parser import parse
from crosscorrelation import find_pixels_shifts
from colormap import colormap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="Liberation Sans")
import copy as cp
import numpy as np
from copy import deepcopy

print "Executing MAIN..."

PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
IMUU_reduced_data = [data.uson for data in IMUU_reduced_data]
#colormap(PIR_data, IMUU_reduced_data, LOG_inflated_data, -1,
#         save_fig=False, smooth=False)

# thresholding on derivative
def thresholding(thresholdu=2.9, thresholdl = -2.9, plotfig=False):

    Uson = cp.deepcopy(IMUU_reduced_data)
    Uson_grad = [y-x for x, y in zip(Uson[:-1], Uson[1:])] 
    Veh = []
    for x in Uson_grad:
        if x < thresholdl:
            Veh.append(1) # Vehicle enters
        elif x > thresholdu:
            Veh.append(-1) # Vehicle leaves
        else:
            Veh.append(0)
    #plt.figure()
    #plt.plot(Uson, label="Ultrasonic signal")
    #plt.plot(Uson_grad, label="Differentiated signal")
    #plt.legend()
    #plt.show()

    #print "Estimated total count: ", np.sum(np.abs(Veh))/2
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
                #print "Missing exit condition at index:", idx
                pass
            else:
                exit = idx
                count.append([enter, exit])
                idx += 1
        elif Veh[idx] == -1:
            #print "Missing enter condition at index:", idx
            idx += 1
        else:
            idx += 1

    #print "Alternative estimation: ", len(count)

    cumulat = [0]*len(Veh)
    for x in count:
        cumulat[x[1]] = 1
    single = deepcopy(cumulat[0:15000])
    for i in range(len(cumulat)-1):
        cumulat[i+1] += cumulat[i]
    Log = cp.deepcopy(LOG_inflated_data)
    cLog = np.array(Log)/10.
    for i in range(len(cLog)-1):
        cLog[i+1] += cLog[i]
    Log = [x/10 for x in Log[0:15000]]

    combined_count = []
    for est, log in zip(single, Log):
        if est == log:
            combined_count.append(0)
        else:
            if est > log:
                combined_count.append(est)
            else:
                combined_count.append(-log)

    measure = 0
    false_pos = 0
    false_neg = 0
    errors = []
    idx = 0
    while idx < len(combined_count):
        if combined_count[idx] != 0:
            measure = combined_count[idx]
            if idx+1 < len(combined_count):
                idx += 1
            else:
                break
            while combined_count[idx] == 0:
                if idx+1 < len(combined_count):
                    idx += 1
                else:
                    break
            if combined_count[idx] == measure:
                if combined_count[idx] == 1:
                    false_pos += 1
                    errors.append([1, idx])
                    #print "False Positive:", idx
                else:
                    false_neg += 1
                    errors.append([-1, idx])
                    #print "False Negative:", idx
            else:
                idx += 1
        else:
            idx += 1
    
    if plotfig:
        errors_timeseries = [0]*len(combined_count)
        for e in errors:
            errors_timeseries[e[1]] = e[0]

        plt.figure()
        #plt.plot(combined_count)
        #plt.plot(Log)
        #plt.plot(single)
        plt.plot(errors_timeseries)
        plt.show()
    return [false_pos, false_neg, false_pos+false_neg]

side = len(np.arange(0.1,5,0.1))
false_pos = -np.ones([side, side])
false_neg = -np.ones([side, side])
false_all = -np.ones([side, side])
for upper_threshold in np.arange(0.1,5,0.1):
    for lower_threshold in np.arange(0.1,5,0.1):
        print "Searching:", [upper_threshold, lower_threshold]
        ret = thresholding(upper_threshold, lower_threshold)
        upper = int(round(upper_threshold*10 - 1))
        lower = int(round(lower_threshold*10 - 1))
        print "Indexing:", [upper, lower]
        print "Falses:", ret
        false_pos[upper][lower] = ret[0]
        false_neg[upper][lower] = ret[1]
        false_all[upper][lower] = ret[2]
        if false_pos[upper][lower] == -1:
            print "___________________________Stop"

upper = np.arange(0.1,5,0.1)
lower = np.arange(0.1,5,0.1)
upper, lower = np.meshgrid(upper, lower)

#print "Minimum false postive %d" % np.min(false_pos)
#np.savetxt("false_positive.csv", false_pos, delimiter=',')
#print "Minimum false negative %d" % np.min(false_neg)
#np.savetxt("false_negative.csv", false_neg, delimiter=',')
#print "Minimum false all %d" % np.min(false_all)
#np.savetxt("false_all.csv", false_all, delimiter=',')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(upper, lower, false_pos, rstride=1, cstride=1, cmap=cm.coolwarm,
                 linewidth=0, antialiased=False)
ax1.set_xlabel("Upper Threshold (m/s)")
ax1.set_ylabel("Lower Threshold (m/s)")
ax1.set_zlabel("False Positive")
plt.title("False Positive")
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(upper, lower, false_neg, rstride=1, cstride=1, cmap=cm.coolwarm,
                 linewidth=0, antialiased=False)
ax2.set_xlabel("Upper Threshold (m/s)")
ax2.set_ylabel("Lower Threshold (m/s)")
ax2.set_zlabel("False Negative")
plt.title("False Negative")
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(upper, lower, false_all, rstride=1, cstride=1, cmap=cm.coolwarm,
                 linewidth=0, antialiased=False)
ax3.set_xlabel("Upper Threshold (m/s)")
ax3.set_ylabel("Lower Threshold (m/s)")
ax3.set_zlabel("All Falses")
plt.title("All Falses")
plt.show()


if False:
    plt.figure()
    #plt.plot(Uson, label="Ultrasonic signal")
    #plt.plot(Log, label="Speed log")
    plt.plot(cLog[0:15000], label="Ground truth")
    #plt.plot(Veh, label="Uson derivative")
    plt.plot(cumulat[0:15000], label="Estimates")
    plt.xlabel("Time (0.125 sec)")
    plt.ylabel("Traffic (Veh)")
    plt.title("Cumulative Traffic")
    plt.legend()
    plt.show()


    accuracy = [x/y for x,y in zip(cumulat, cLog)]
    plt.figure()
    plt.plot(accuracy[0:15000], label="Estimation accuracy")
    plt.plot([1.05]*15000, label="105%")
    plt.plot([1]*15000, label="100%")
    plt.plot([0.95]*15000, label="95%")
    plt.xlabel("Time (0.125 sec)")
    plt.ylabel("Estimated traffic/ground truth")
    plt.title("Cumulative Accuracy of Vehicle Counting")
    plt.legend()
    plt.show()
