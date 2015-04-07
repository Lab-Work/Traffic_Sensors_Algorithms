'''
TODO:
1) Documentation
2) Citation 
'''

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time



def parseTemp():
	try:
	    l = file.next()
	    #print 'Parse line:'
	    #print l
	    #print len(l)
	    
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
	except:
		print '[ERROR] I want to stop animation, but I do not know how.'



def updatefig(*args):
    global im
    im.set_array(parseTemp())
    return im,



def animate():
    global im
    global ani
    fig,ax = plt.subplots()
    plt.title('Thermal Rendering of the Infraed Signal')

    #im = plt.imshow(parseTemp(),cmap=plt.get_cmap('jet'),interpolation='nearest')
    im = plt.imshow(parseTemp(),cmap=plt.get_cmap('jet'))
    plt.colorbar(im,orientation="horizontal")
    ax.grid(True)

    ani = animation.FuncAnimation(fig, updatefig, interval=200, blit=True, repeat = False, save_count=10)
    plt.show()




if __name__ == '__main__':
    #filename = raw_input('Enter the playback text file name: ')
    filename = 'test_20150310-185259.txt'
    with open(filename, 'r') as file:
        for i in range(5):
            file.next()
        animate()
    #Set up formatting for the movie files
    Writer = animation.writers['mencoder']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=3600)
    ani.save('thermal_1Hzfps5.mp4', writer=writer)
