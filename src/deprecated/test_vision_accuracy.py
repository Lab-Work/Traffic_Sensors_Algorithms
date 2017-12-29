import numpy as np
import matplotlib.pyplot as plt

# data format:
# [t_in, t_out, speed (mph), distance (m), ...]
v11 = np.load('../workspace/0530_2017/labels_v11_post.npy')
m_v11 = np.load('../workspace/0530_2017/labels_v11_manual_post.npy')

speeds = []
for i in xrange(len(m_v11)):
    speeds.append([v11[i][2], m_v11[i][2]])

speeds = np.asarray(speeds)
err = speeds[:,0] - speeds[:,1]


# comptue rmse
rmse = np.sqrt( np.sum(err**2)/len(err) )

# std
mean = np.mean(err)
std = np.sqrt( np.sum((err-mean)**2)/len(err) )

# Compute the average error
print('Vision accuracy (mph):  bias,     std,       rmse')
print('                      {0:.3f},      {1:.5f},    {2:.5f}'.format(np.mean(err), std, rmse))

# plot the distribution
n, bins, patches = plt.hist(err, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()

