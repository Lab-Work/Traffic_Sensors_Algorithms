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
    - If the data is perfectly normally distributed, then want to include data within (-sigma, sigma) interval
"""

# ============================================================================
# First case: what if the data is perfectly normally distributed.
d = np.random.normal(0,1,100000)
# d = np.random.uniform(-1,1,200000)
# noise = np.random.uniform(-5,5,30)

# d = np.concatenate([d, noise])
# gmm = GaussianMixture()

# w = 0.5
# # ratio = 2.268     # tested value is 2.268 for perfect Gaussian to converge to (-2*sigma, 2*sigma) interval
# exp_r = 2.5
# con_r = 1.73
# print('For Gausian distribution:')
# for i in xrange(0,20):
#     print('   {0}, w: {1}'.format(i, w))
#     # get the data points within in the tolerance
#     ix = (d>=-w) & (d<=w)
#     r = gmm.fit(d[ix,np.newaxis])
#     sig = np.sqrt(r.covariances_[0,0])

#     # set new tolerance
#     if w > 1:
#         w = sig[0]*con_r
#     else:
#         w = sig[0]*exp_r

w = 0.5

# for uniform distribution: theoretical threshold in sqrt(3)
# exp_r = 1.8
# con_r = 1.7
# for normal distribution: theoretical threshold is 2.27369437677
exp_r = 2.27369437678
con_r = 2.27369437676

print('\nFor STD:')
for i in xrange(0,20):
    print('   {0}, w: {1}'.format(i, w))
    # get the data points within in the tolerance
    ix = (d>=-w) & (d<=w)
    sig2 = np.std(d[ix])

    # set new tolerance
    # set new tolerance
    if w > 1:
        w = sig2*con_r
    else:
        w = sig2*exp_r

# Compute the thoeretical ratio for perfect Guausian
#   - Goal, given truncated guassian (-2*sigma, 2*sigma), the computed tolerance is still (-2*sigma, 2*sigma)
#   - See how to compute the std for truncated guassian in https://en.wikipedia.org/wiki/Truncated_normal_distribution
Z = 0.9545
alpha = -2.0
beta = 2.0
phi_alpha = np.exp(-0.5*(alpha**2))/np.sqrt(2*np.pi)
phi_beta = np.exp(-0.5*(beta**2))/np.sqrt(2*np.pi)
std = np.sqrt( 1 + (alpha*phi_alpha-beta*phi_beta)/Z - np.power( (phi_alpha-phi_beta)/Z ,2) )
print('Truencated guassian [-2, 2] std: {0}'.format(std))
print('ratio = 2/std: {0}'.format(2.0/std))

n, bins, patches = plt.hist(d, 50, normed=1, facecolor='green', alpha=0.75)
plt.plot(bins, mlab.normpdf(bins, 0, 1), color='g', linestyle='--', label='true', linewidth=2)
# plt.plot(bins, mlab.normpdf(bins, 0, sig), color='r', linestyle='--', label='fitted', linewidth=2)
plt.plot(bins, mlab.normpdf(bins, 0, sig2), color='g', linestyle='--', label='fitted', linewidth=2)
plt.xlim([-2,2])

plt.show()



# ============================================================================





