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

PIR_data, IMUU_reduced_data, LOG_inflated_data = parse()
#print IMUU_reduced_data[0]
for data in IMUU_reduced_data:
    try:
        data.uson
    except:
        print data
Uson = [data.uson for data in IMUU_reduced_data[0:]]
Log = deepcopy(LOG_inflated_data)
plt.figure()
plt.plot(Uson)
plt.plot(Log)
plt.axhline(10)
plt.show()

if True:
    PIR1 = np.array(PIR_data[0][0:64]).reshape((4,16))
    PIR2 = np.array(PIR_data[0][64:128]).reshape((4,16))
    PIR3 = np.array(PIR_data[0][128:192]).reshape((4,16))
    Uson = [data.uson for data in IMUU_reduced_data[0:100]]
    Mag = IMUU_reduced_data[0].mag

    fig = plt.figure(figsize=(60, 8), dpi=100)
    ax11 = fig.add_subplot(231)
    ax11.imshow(PIR1, interpolation='nearest',vmin=30,vmax=50)
    ax11.set_title("PIR 1")
    ax12 = fig.add_subplot(232)
    ax12.imshow(PIR2, interpolation='nearest',vmin=30,vmax=50)
    ax12.set_title("PIR 2")
    ax13 = fig.add_subplot(233)
    ax13.imshow(PIR3, interpolation='nearest',vmin=30,vmax=50)
    ax13.set_title("PIR 3")



    true_count = True
    count = True
    det_error = None

    true_speed = 50
    speed = 51
    est_error = None

    ax21 = fig.add_subplot(234)
    ax21.text(0,0.8,"Detected vehicles (veh): %s" % count, fontsize=16) 
    ax21.text(0,0.7,"Observed vehicles (Veh): %s" % true_count,fontsize=16)
    ax21.text(0,0.6,"Detection Error: %s" % det_error, fontsize=16)
    ax21.text(0,0.4,"Estimated Speed (mph): %.2f" % speed, fontsize=16)
    ax21.text(0,0.3,"Observed Speed (mph): %.2f" % true_speed, fontsize=16)
    ax21.text(0,0.2,"Estimation Error: %s" % est_error, fontsize=16)
    ax21.axis("off")

    ax22 = fig.add_subplot(235)
    ax22.plot(np.arange(-len(Uson)/2, len(Uson)/2),Uson)
    ax22.axhline(10, color='k', linestyle='--')
    ax22.axvline(0, color='k', linestyle='--')
    ax22.set_xlim([-len(Uson)/2, len(Uson)/2])
    ax22.set_xlabel("Time (0.125 sec)")
    ax22.set_title("Distance (m)")
    ax22.set_title("Utrasonic")

    ax23 = fig.add_subplot(236, projection='3d')
    ax23.scatter(Mag[0], Mag[1], Mag[2])
    ax23.set_xlabel("X direction")
    ax23.set_ylabel("Y direction")
    ax23.set_zlabel("Z direction")
    ax23.set_title("Magnetometer")
    plt.show()

