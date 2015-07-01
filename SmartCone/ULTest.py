import csv
import numpy
import matplotlib.pyplot as plt

with open('database/ULTest.csv','r') as file:
    reader = csv.reader(file)
    FILES = numpy.array([line for line in reader])
    FILES = numpy.array([[float(line[1]), float(line[2]), float(line[3]), float(line[4])] for line in FILES[0:1000]])
    
    #print len(FILES[0])
    plt.figure()
    plt.plot(FILES[:,0],FILES[:,1],'*',FILES[:,2],FILES[:,3],'.')
    plt.figure()
    plt.plot(FILES[:,1],FILES[:,3],'.')
    plt.show()
