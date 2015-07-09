import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True) # This will create LaTex style plotting.

'''
#The raw file needs to be moderately processed to omit text before numerical analysis
with open('database/ULTest.csv','r') as src:
    reader = csv.reader(src)
    with open('database/ULTest_2.csv','w') as dest:
        writer = csv.writer(dest)
        for line in reader:
            if line[0] == '#':
                writer.writerow(line)

with open('database/ULTest_2.csv','r') as dest:
    reader = csv.reader(dest)
    for line in reader:
        if len(line) != 6:
            print line
'''

def timeFormat(datenum):
    return datetime.fromtimestamp(float(datenum))

ultraTime = []
laserTime = []
ultra = []
laser = []
count = []
with open('database/ULTest_2.csv','r') as dest:
    reader = csv.reader(dest)
    for line in reader:
        ultraTime.append(float(line[1])/60000)
        ultra.append(float(line[2]))
        laserTime.append(float(line[3])/60000)
        laser.append(float(line[4]))
        count.append(float(line[5]))

ultra = np.array(ultra) - np.average(ultra)
laser = np.array(laser)

laserMode = max(laser)
print laserMode

for i in range(len(laser)):
    if laser[i] > 10:
        laser[i] = 0

distance = np.array([i**2 + j**2 for i, j in zip(ultra, laser)])

plt.figure()
plt.plot(ultraTime, -ultra, label='Ultrasonic')
plt.plot(laserTime, laser, label='Laser')
#plt.plot(ultraTime, distance, label='$U^2 + L^2$')
plt.xlabel('Machine Time')
plt.ylabel('Relative Signal Maganitude')
plt.title('Comparison of Laser and Ultrasonic Sensor Signals 06/25/15')
plt.legend()
#plt.show()


COUNT = []
for i in range(len(distance)):
    if distance[i] > 20:
        COUNT.append(1)
    else:
        COUNT.append(0)

BASKET = []
TMP = []
for i in range(len(COUNT)):
    if i == 0:
        TMP = [COUNT[i]]
    if COUNT[i] == TMP[0]:
        TMP.append(COUNT[i])
    else:
        BASKET.append(TMP)
        TMP = [COUNT[i]]

for SPK in BASKET:
    if len(SPK) > 2 and SPK[0] == 1:
        for i in range(len(SPK)):
            if i != 0:
                SPK[i] = 0
    elif SPK[0] == 1:
        for i in range(len(SPK)):
            SPK[i] = 0
#print BASKET

BASKET = np.array([i for SUB in BASKET for i in SUB])
BASKET = np.cumsum(BASKET)

logTime = []
logSPK = []
with open('data/log.txt', 'r') as log:
    reader = csv.reader(log)
    for i in range(3): # Align starting time
        reader.next()
    for line in reader:
        logTime.append(datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S.%f'))
        logSPK.append(float(line[2]))

logSPK = np.array([i+25 for i in logSPK])
print len(logSPK)
print max(BASKET)

logTimestamp = [(i-logTime[0]) for i in logTime]
logTimestamp = [i.total_seconds()/60 + i.microseconds/60000000 for i in logTimestamp]

plt.figure()
plt.plot(logTimestamp, logSPK, label='Ground truth')
plt.plot(ultraTime[:-1], BASKET, label='Computed count')
plt.legend()
plt.xlabel('Machine Time (min)')
plt.ylabel('Cumulative Traffic')
plt.title('Computed Cumulative Traffic Diagram')
plt.show()


'''
counter = 0
for SPK in  BLASKET:
    if len(SPK) > 2 and SPK[0] == 1:
        counter += 1
        print counter
'''

