'''
TODO:
1) Documentation
2) Citation 
'''

from serial import Serial
from serial.tools import list_ports
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
from threading import Thread
import time


def serial():
    global l
    #baud = 9600
    baud = 115200

    #Scan USB ports, and find the serial connection to the Arduino
    port_list = list_ports.comports()
    for line in port_list:
	if('usb' in line[2].lower()):
            chosenPort = line[0]
    ser = Serial(chosenPort,baud)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    myfile = open("test_"+timestamp+".txt", "w") 

    while(True):
        try:
            l = ser.readline()
            print datetime.now()
            print(l)
            myfile.write(l)
        except:
            myfile.close()
            print '[WARNING] Exception found'



def parseTemp(l):
    #print 'Parse line:'
    print l
    
    L = l[2:2+3*64]
    T = []
    for i in range(len(L)):
        if i%3 == 0 and i != 0:
            t = float(L[i-3:i])/10
            T.append(t)
    t = float(L[len(L)-3:len(L)])/10
    T.append(t)

    T = np.array(T).reshape((4,16),order='F')
    T = T
    T0 = float(l[2+3*64:2+3*64+3])/10
    count = int(l[2+3*65:2+3*65+2])

    #print 'Temperature readings:', T
    #print 'Ambient temperature:', T0
    #print 'Signal counts:', count
    return T



def updatefig(*args):
    global im
    im.set_array(parseTemp(l))
    return im,



def animate():
    global l
    global im
    fig,ax = plt.subplots()
    plt.title('Random Thermal Signal Teexcept under except pythonst')

    #im = plt.imshow(parseTemp(l),cmap=plt.get_cmap('jet'),interpolation='nearest')
    im = plt.imshow(parseTemp(l),cmap=plt.get_cmap('jet'))
    plt.colorbar(im,orientation="horizontal")
    ax.grid(True)

    ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True, save_count=2000)
    plt.show()

    #Set up formatting for the movie files
    #Writer = animation.writers['mencoder']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=3600)
    #ani.save('thermal_fps15.mp4', writer=writer)


if __name__ == '__main__':
    global SER
    SER = Thread(target = serial)
    SER.start()
    time.sleep(5)
    Thread(target = animate).start()
