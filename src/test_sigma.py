import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.mixture import GaussianMixture

"""
This script analyze the parameter for expansion and contraction switch.
The process is
    - At each step, run a GMM to the data set within tolerance w_old and get sigma
    - Then depending on the fitting metric, select a new tolerance parameter w_new = f(sigma) to either give the
    expansion (w_new > w_old) or contraction (w_new < w_old) fact to the data set.
    - The iteration stops when there is negligible differences.

Observations:
    - If the data is perfectly normally distributed, then want to include data within 2-sigma interval
"""

# ============================================================================
# First case: what if the data is perfectly normally distributed.
# d = np.random.normal(0,1,100000)
d = np.random.uniform(-1,1,100000)
gmm = GaussianMixture()

w = 1.0
# ratio = 2.268     # theoretical value is 2.268 for perfect Gaussian to converge to 2-sigma interval
ratio = 2.0
print('For Gausian distribution:')
for i in xrange(0,20):
    print('   {0}, w: {1}'.format(i, w))
    # get the data points within in the tolerance
    ix = (d>=-w) & (d<=w)
    r = gmm.fit(d[ix,np.newaxis])
    sig = np.sqrt(r.covariances_[0,0])

    # set new tolerance
    w = sig[0]*ratio

# n, bins, patches = plt.hist(d, 50, normed=1, facecolor='green', alpha=0.75)
# plt.plot(bins, mlab.normpdf(bins, 0, 1), color='g', linestyle='--', label='true', linewidth=2)
# plt.plot(bins, mlab.normpdf(bins, 0, sig), color='r', linestyle='--', label='fitted', linewidth=2)
#
# plt.show()







