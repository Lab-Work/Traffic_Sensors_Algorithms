'''/////////////////////////////////////////////////////////////////////////////
@Author: Fangyu Wu
@Date created: 08/11/2015

A short script that synchronizes all sensor data to observe its
multi-dimensional spatial distribution over all sensors data collected on
06/25/2015 by Yanning Li, Will Barbour, and Raphi Stern.
/////////////////////////////////////////////////////////////////////////////'''

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import csv
from collections import namedtuple
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True) # This will create LaTex style plotting.

# A short format routine that converts unix timestamp to python datetime
def timestamp_to_datetime(datenum):
    return datetime.fromtimestamp(float(datenum))

# Import PIR raw data to a struct (namedtuple)
PIR_raw = open("PIR.csv", 'r')
PIR_csv = csv.reader(PIR_raw)
# Store each measurement into an namedtuple
PIR_instance = namedtuple(
    "PIR_instance",
    """begin
       millis1 pir1 amb1
       millis2 pir2 amb2
       millis3 pir3 amb3"""
)

PIR_data = []
for line in PIR_csv:
    PIR_data.append(
        PIR_instance(
            timestamp_to_datetime
            (float(line[0])),                   # Start time
            float(line[1]),                     # Millis time (L)
            [float(x) for x in line[2:66]],     # 64 pixels temperature (L)
            float(line[66]),                    # Ambient temperature (L)
            float(line[68]),                    # Millis time (M)
            [float(x) for x in line[69:133]],   # 64 pixels temperature (M)
            float(line[133]),                   # Ambient temperature (M)
            float(line[135]),                   # Millis time (R)
            [float(x) for x in line[136:200]],  # 64 pixels temperature (R)
            float(line[200])                    # Ambient temperature (R)
        )
   )
PIR_raw.close()


# Import Ulson&Laser raw data to a struct (namedtuple)
UL_raw = open("UL.csv", 'r')
UL_csv = csv.reader(UL_raw)
UL_instance = namedtuple(
    "UL_instance",
    """begin ulson laser"""
)

UL_data = []
for line in UL_csv:
    UL_data.append(
        UL_instance(
            float(line[3])/60000,   # Instance time
            12-float(line[2]),         # Ultrasonic measurement
            float(line[4]),         # Laser measurement
        )
    )
UL_raw.close()

# Import Acc/Gyro/Mag raw data to a struct (namedtuple)
IMU_raw = open("IMU.csv", 'r')
IMU_csv = csv.reader(IMU_raw)
IMU_instance = namedtuple(
    "IMU_instance",
    "begin mag acc gyr"
)

IMU_data = []
for line in IMU_csv:
    IMU_data.append(
        IMU_instance(
            timestamp_to_datetime(float(line[0])),
            # Instance time
            [float(line[1].replace("[","")),
             float(line[2]),
             float(line[3].replace("]",""))],
            # Magnetometer measurement
            [float(line[4].replace("[","")),
             float(line[5]),
             float(line[6].replace("]",""))],
            # Accelerometer measurement
            [float(line[7].replace("[","")),
             float(line[8]),
             float(line[9]),
             float(line[10].replace("'","")),
             float(line[11].replace("'","")),
             float(line[12].replace("'","").replace("]",""))]
            # Gyroscope measurement
        )
    )
IMU_raw.close()

# Import speed log to a struct (namedtuple)
LOG_raw = open("LOG.csv", 'r')
LOG_csv = csv.reader(LOG_raw)
LOG_instance = namedtuple(
    "LOG_instance",
    """begin speed count"""
)

LOG_data = []
for line in LOG_csv:
    LOG_data.append(
        LOG_instance(
            datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S.%f'), # Instance time
            line[1],                                            # Speed
            int(line[2]),                                       # Count
        )
    )
LOG_raw.close()

PIR_timestamp = [t.begin-PIR_data[0].begin for t in PIR_data]
PIR_timestamp = [t.total_seconds()/60 + t.microseconds/60000000
                 for t in PIR_timestamp]
UL_timestamp = [t.begin-UL_data[0].begin+0.15 for t in UL_data]
IMU_timestamp = [t.begin-IMU_data[0].begin for t in IMU_data]
IMU_timestamp = [t.total_seconds()/60 + t.microseconds/60000000
                 for t in IMU_timestamp]
LOG_timestamp = [t.begin-LOG_data[0].begin for t in LOG_data]
LOG_timestamp = [-2.23 + t.total_seconds()/60 + t.microseconds/60000000
                 for t in LOG_timestamp]

PIR_idx = range(len(PIR_data))
from bisect import bisect_left
UL_idx = [bisect_left(UL_timestamp, t) for t in PIR_timestamp]
IMU_idx = range(len(PIR_data))
LOG_idx = [bisect_left(PIR_timestamp, t) for t in LOG_timestamp]
LOG_data_new = [0]*len(PIR_timestamp)
for i in LOG_idx:
    try:
        LOG_data_new[i] = 8
    except:
        pass

norm_avg = []
for i in range(64):
    norm_avg.append(np.mean([t.pir1[i] for t in PIR_data]))
for i in range(64):
    norm_avg.append(np.mean([t.pir2[i] for t in PIR_data]))
for i in range(64):
    norm_avg.append(np.mean([t.pir3[i] for t in PIR_data]))
norm_avg.append(np.mean([t.ulson for t in UL_data]))
norm_avg.append(np.mean([t.ulson for t in UL_data]))
norm_avg.append(np.mean([t.ulson for t in UL_data]))
norm_avg.append(np.mean([t.ulson for t in UL_data]))
norm_avg.append(np.mean([t.ulson for t in UL_data]))
norm_avg.append(np.mean([t.ulson for t in UL_data]))
norm_avg.append(0)
norm_avg.append(0)
norm_avg.append(0)
norm_avg.append(0)
norm_avg.append(0)
norm_avg.append(0)

norm_std = []
for i in range(64):
    norm_std.append(np.std([t.pir1[i] for t in PIR_data]))
for i in range(64):
    norm_std.append(np.std([t.pir2[i] for t in PIR_data]))
for i in range(64):
    norm_std.append(np.std([t.pir3[i] for t in PIR_data]))
norm_std.append(np.std([t.ulson for t in UL_data]))
norm_std.append(np.std([t.ulson for t in UL_data]))
norm_std.append(np.std([t.ulson for t in UL_data]))
norm_std.append(np.std([t.ulson for t in UL_data]))
norm_std.append(np.std([t.ulson for t in UL_data]))
norm_std.append(np.std([t.ulson for t in UL_data]))
norm_std.append(1)
norm_std.append(1)
norm_std.append(1)
norm_std.append(1)
norm_std.append(1)
norm_std.append(1)

def check_colormap(start=0, end=35000, save_plot=False):
    colormap = []
    for x, y, z in zip(PIR_data[start:end], UL_idx[start:end],
                       LOG_data_new[start:end]):
        colormap.append((np.array(x.pir1 + x.pir2 + x.pir3 +
                                  [UL_data[y].ulson]*6 + [z]*6)
                         - np.array(norm_avg)) /
                        np.array(norm_std))
    colormap = np.array(colormap)
    #colormap = colormap.squeeze()
    colormap = np.transpose(colormap)
    plt.figure(figsize=(40,18), dpi=250)
    plt.imshow(colormap, origin="lower", cmap=plt.get_cmap("jet"), aspect="auto",
               interpolation="nearest", vmin=-2, vmax=8)
    #plt.colorbar(orientation="horizontal", fontsize=32)
    plt.title("Sample Colormap from Data Collected on 06/25/15", fontsize=32)
    plt.ylabel("Normalized Signal from PIR and Ulson", fontsize=32)
    plt.xlabel("Elapsed time /0.125 sec", fontsize=32)
    if (save_plot):
        plt.savefig("color_maps/"+"{:06}".format(start))
        plt.close()

for t in range(35000-2000):
    check_colormap(t,t+2000,True)

check_2d_scatter = False
if (check_2d_scatter):
    plt.figure()
    plt.scatter([t.pir1[24] for t in PIR_data[1000:35000]],
                [UL_data[i].ulson for i in UL_idx[1000:35000]],
                marker='.')
    plt.xlabel("PIR(L) pixel 24 (temperature)")
    plt.ylabel("Ultrasonic (distance)")
    plt.title("Left PIR 24th Pixel v.s. Ultrasonic")
    plt.savefig("2d_scatter.png")

check_3d_scatter = False
if (check_3d_scatter):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([t.pir1[24] for t in PIR_data[10000:35000]],
               [t.pir1[48] for t in PIR_data[10000:35000]],
               [UL_data[i].ulson for i in UL_idx[10000:35000]],
               marker='.')
    ax.set_xlabel("PIR(L) pixel 24")
    ax.set_ylabel("PIR(L) pixel 48")
    ax.set_zlabel("Ultrasonic")
    ax.set_title("Left PIR 24th, 48th Pixels v.s. Ultrasonic")
    plt.savefig("3d_scatter.png")

check_time_series = False
if (check_time_series):
    plt.figure()
    plt.plot(PIR_timestamp[0:35000],
             [t.pir1[24] for t in PIR_data[0:35000]],
             label = "PIR(L) pixel 24")
    plt.plot([UL_timestamp[i] for i in UL_idx[0:35000]],
             [UL_data[i].ulson for i in UL_idx[0:35000]],
             label = "Ultrasonic")
    plt.legend()
    plt.title("Time Synchronization Check")
    #plt.savefig("timeseries.png")

check_time = False
if (check_time):
    plt.figure()
    plt.plot(PIR_timestamp, 'ro')
    plt.plot(UL_timestamp, 'b')
    plt.xlabel("Number of Measurement")
    plt.ylabel("Eplased Time (min)")
    plt.title("Measurement Time Coherency Inspection (Slope Stands for 1/Frequency)")
    plt.plot(IMU_timestamp, 'go')
    plt.savefig("Erroneous_Ulson_Laser_Frequency.png")

plt.show()
