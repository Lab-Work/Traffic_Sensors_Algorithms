import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=18)
from scipy.signal import savgol_filter

# Credit to cji on Stackoverflow
def find_between(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

SRC = sys.argv[1]
freq = int(find_between(SRC, "fov", "hz"))
scale=64

pir = np.load(SRC)
pir = np.asarray([frame for frame in pir[:,1]])
pir_shape = pir.shape
pir = np.asarray([frame.flatten() for frame in pir])
median = np.median(pir, axis=0)
std = np.std(pir, axis=0)
pir = (pir - median)/std*255
pir = abs(savgol_filter(pir, 29, 2, axis=0))
pir[pir > 255] = 255

plt.figure()
for w in range(16):
    for h in range(4):
        plt.plot(pir[:,16*h+w])
plt.show()


pir = pir.reshape(pir_shape)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(SRC.replace("npy", "avi"),fourcc, freq, (16*scale,4*scale))
pir_cam = []
for frame in pir:
    #print frame
    img = np.asarray(frame).astype(np.uint8)
    img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    out.write(img)
    cv2.imshow("PIR Cam", img)
    cv2.waitKey(10)
out.release()
cv2.destroyAllWindows()
