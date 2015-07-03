import csv
import numpy as np
import matplotlib.pyplot as plt

'''
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
uT = []
lT = []
ultra = []
laser = []
count = []
with open('database/ULTest_2.csv','r') as dest:
    reader = csv.reader(dest)
    for line in reader:
        uT.append(line[1])
        ultra.append(line[2])
        lT.append(line[3])
        laser.append(line[4])
        count.append(line[5])

print uT[0:10]
print len(uT)
print len(lT)
print len(ultra)
print len(laser)
print len(count)
