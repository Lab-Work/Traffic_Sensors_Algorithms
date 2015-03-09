import matplotlib.pyplot as plt
from pylab import *
A = zeros((4,16))
fig = plt.figure()
fig.show()
i = 0
j = 0
while True:
    try:
        i += 1
        j += 1
        A[(i)%4,(j)%16] += 1
        imshow(A, interpolation='nearest')
        fig.canvas.draw()
    except KeyboardInterrupt:
        break
