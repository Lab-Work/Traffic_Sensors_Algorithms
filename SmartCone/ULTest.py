import csv
import numpy as np
import matplotlib.pyplot as plt

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

ultraTime = []
laserTime = []
ultra = []
laser = []
count = []
with open('database/ULTest_2.csv','r') as dest:
    reader = csv.reader(dest)
    for line in reader:
        ultraTime.append(float(line[1]))
        ultra.append(float(line[2]))
        laserTime.append(float(line[3]))
        laser.append(float(line[4]))
        count.append(float(line[5]))

ultra = np.array(ultra)
ultra = (ultra - np.average(ultra)) / np.std(ultra)

laser = np.array(laser)
laser = (laser - np.average(laser)) / np.std(laser)

SPK = 0
for i in range(len(ultra)):
    if ultra[i] > -2:
        laser[i] = 0.9

for i in range(len(ultra)-1):
    if ultra[i+1] - ultra[i] < -1.5:
        SPK += 1
print SPK

plt.figure()
plt.plot(ultraTime, ultra)
plt.plot(laserTime, laser)
plt.show()
