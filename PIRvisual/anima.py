import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import time

N = 500
start = time.time()

X,Y = np.meshgrid(np.arange(5), np.arange(17))
Z = np.random.rand(16,4)
#Z = Z[:-1, :-1]

ax = plt.subplot(111)
thermal = plt.pcolormesh(X,Y,Z)
#thermal = pl.imshow(Z,interpolation='nearest')
plt.colorbar()

plt.ion()
plt.show()

i = 0
j = 0
for frame in np.arange(N):
    try:
       # i += 1 
       # j += 1
       # Z[(i)%16, (j)%4] += 1
       # print Z
        Z = np.random.rand(16,4)
       # Z = Z[:-1, :-1]
       # print Z
        thermal.set_array(Z.ravel())
        for y in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                plt.text(x + 0.5, y + 0.5, '%.4f' % Z[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                     )
       # plt.title('Frame: %.2f'%frame)
        plt.draw()
    except KeyboardInterrupt:
        break
print 'FPS', N/(time.time()-start)
print '# of frames:', N

plt.ioff()
plt.show()



