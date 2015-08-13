import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

time = []
mag = []
acc =[]
gyr = []
with open('IMU_Jun_25.csv.bak', 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        time.append(float(line[0]))
        mag.append(line[1].replace("[","")+line[2]+line[3].replace("]",""))
        acc.append(line[4].replace("[","")+line[5]+line[6].replace("]",""))
        gyr.append(line[7].replace("[","")+line[8]+line[9]+
                   line[10].replace("'","")+
                   line[11].replace("'","")+
                   line[12].replace("'","").replace("]",""))

#print time[0], len(time)
print mag[0], len(mag)
mag = [line.split() for line in mag]
print mag[0], len(mag)

fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
xCoor = [float(pt[0]) for pt in mag] 
yCoor = [float(pt[1]) for pt in mag]
zCoor = [float(pt[2]) for pt in mag]
ax.scatter(xCoor,yCoor,zCoor)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.figure(2)
plt.plot(time,xCoor)



'''
print acc[0], len(acc)
acc = [line.split() for line in acc]
print acc[0], len(acc)

fig = plt.figure(3)
ax = fig.add_subplot(111,projection='3d')
xCoor = [float(pt[0]) for pt in acc] 
yCoor = [float(pt[1]) for pt in acc]
zCoor = [float(pt[2]) for pt in acc]
ax.scatter(xCoor,yCoor,zCoor)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.figure(4)
plt.plot(time,xCoor)

plt.show()

#print acc[0], len(acc)
#print gyr[0], len(gyr)
'''
