import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import hough_line, hough_line_peaks
from mpl_toolkits.mplot3d import Axes3D
import sys
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import glob

"""
This script is used to test the original hough line transform and the cluster algorithm
"""

def main(argv):

    folder = '../figs/hough_realdata/'

    files = glob.glob(folder+'*.npy')

    print files

    for f in files:
        # img = generate_image()
        img = get_image(f)

        # plt.imshow(1-img, cmap=plt.cm.gray, interpolation='nearest')
        # plt.imshow(1-img, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')

        # the hough line transform
        hspace, angles, dists = hough_line(img)
        # plot_hspace(hspace, thres=np.max(hspace)/2)
        # plot_hspace(hspace, thres=None)

        # hough peak
        hspace_max, peak_angles, peak_dists = hough_line_peaks(hspace, angles, dists, num_peaks=3)
        # print peak_angles

        # clustering
        # c_angles, c_dists = hough_cluster(img, hspace, angles, dists,
        #                                   thres=np.max(hspace)*1/2,
        #                                   num_cluster=2, plot=True)

        plot_hough(img, peak_angles, peak_dists, hspace, angles, dists)
    plt.show()

def generate_image():
    """
    This function generates a image
    :return:
    """
    # generate a custom image
    img = np.zeros((100, 150), dtype=bool)

    for row in range(0, 100):
        # add first band
        for col in range(int(round(50+0.5*row)), int(round(60+0.5*row))):
            v = np.random.uniform()
            if v <= 0.4:
                img[row, col] = 1

        # add second band
        for col in range(int(round(50+0.3*row)), int(round(60+0.3*row))):
            v = np.random.uniform()
            if v <= 0.4:
                img[row, col] = 1

        # negative line
        # for col in range(int(round(70-0.3*row)), int(round(80-0.3*row))):
        #     v = np.random.uniform()
        #     if v <= 0.3:
        #         img[row, col] = 1

        # add third line
        # for col in range(int(round(100+0.3*row)), int(round(101+0.3*row))):
        #     v = np.random.uniform()
        #     if v <= 0.9:
        #         img[row, col] = 1

    return img


def get_image(f):

    img = np.load(f)

    return img


def hough_cluster(img, hspace, angles, dists, thres=None, num_cluster=2, plot=False):
    """
    This function clusters the peak values and use cluster to find the peak values.
    :param hspace:
    :param angles:
    :param dists:
    :param thres:
    :param num_cluster:
    :return:
    """

    _hspace = deepcopy(hspace)
    if thres is not None:
        _hspace[_hspace <= thres] = 0

    # plot hspace
    # figure = plt.figure()
    # ax = figure.gca(projection='3d')
    # A, D = np.meshgrid(angles, dists)
    # Z = hspace
    # surf = ax.plot_surface(A, D, Z, cmap=cm.coolwarm)
    # figure.colorbar(surf, shrink=0.5, aspect=5)

    samples = []    # row_idx, col_idx, val
    # make a list of samples: num_samples x num_features, and weight
    for row_idx, row in enumerate(_hspace):
        for col_idx, val in enumerate(row):
            if val != 0:
                samples.append([row_idx, col_idx, val])

    samples = np.asarray(samples)

    # Use k-means to find the center
    # y_pre = KMeans(n_clusters=num_cluster).fit_predict(samples[:,0:2])

    # y_pre = DBSCAN().fit_predict(samples[:,0:2], sample_weight=samples[:,2])
    # y_pre = DBSCAN(eps=2, min_samples=30).fit_predict(samples[:,0:2])

    w = np.mean(samples[:,2])
    # samples[:,2] = w
    y_pre = DBSCAN(eps=1.5, min_samples=8*w).fit_predict(samples[:,0:2], sample_weight=samples[:,2])

    n_clusters_ = len(set(y_pre)) - (1 if -1 in y_pre else 0)

    y_pre = np.asarray(y_pre)
    if plot is True:
        fix, axes = plt.subplots(1,2, figsize=(15, 8))
        # plot the original space
        axes[0].imshow(1-img, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
        axes[0].set_title('Input image')
        # plot the simple hough peak results
        axes[0].autoscale(False)

        # plot the hough space clustering
        axes[1].imshow(_hspace, cmap=plt.cm.bone, interpolation='nearest',
                extent=(np.rad2deg(angles[-1]), np.rad2deg(angles[0]), dists[-1], dists[0]), aspect='auto')
        # axes.imshow(_hspace, cmap=plt.cm.bone, interpolation='nearest')
        axes[1].autoscale(False)
        colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w',
                  'r', 'b', 'g', 'k', 'm', 'y', 'c', 'w']
        axes[1].set_xlabel('Angle (degree)')
        axes[1].set_ylabel('Distance (pixel)')
        axes[1].set_xlim([40,0])
        axes[1].set_ylim([80,20])

    print('Total {0} clusters:'.format(n_clusters_))
    clus_center = []
    for clus_label in range(0, n_clusters_):
        clus = (y_pre == clus_label)
        clus_dist = dists[ samples[clus,0].astype(int) ]
        clus_angle = angles[ samples[clus,1].astype(int) ]
        num_pts = sum(clus)
        center = [np.mean(clus_angle), np.mean(clus_dist)]
        print('--- {0} pt: {1} degree, {2} pixel'.format(num_pts, -np.rad2deg(center[0]), center[1]))

        clus_center.append((num_pts, center))
        if plot is True:
            # plot the cluster in Hough space
            axes[1].scatter(-np.rad2deg(clus_angle), clus_dist, color=colors[clus_label])
            axes[1].scatter(-np.rad2deg(center[0]), center[1], color=colors[clus_label], marker='*', s=100)

    if plot is True:
        if n_clusters_ >=2:
            clus_center.sort(reverse=True)
            for i in range(0,2):
                center = clus_center[i][1]
                # plot the line on original image
                x0 = center[1]/np.cos(-center[0])
                x100 = 100.0*np.tan(-center[0]) + center[1]/np.cos((-center[0]))
                axes[0].plot([x0, x100], [0,100], linewidth=2, color='r')
        else:
            center = clus_center[0][1]
            # plot the line on original image
            x0 = center[1]/np.cos(-center[0])
            x100 = 100.0*np.tan(-center[0]) + center[1]/np.cos((-center[0]))
            axes[0].plot([x0, x100], [0,100], linewidth=2, color='r')


    # sort indeceding order and only return the first two
    if n_clusters_ >2:
        clus_center.sort(reverse=True)
        return [-np.rad2deg(clus_center[i][1][0]) for i in range(0, 2)], [clus_center[i][1][1] for i in range(0,2)]
    else:
        # return two lists: angles and dists
        return [i[1][0] for i in clus_center], [i[1][1] for i in clus_center]




def plot_hspace(hspace, thres=None):

    hspace_thres = deepcopy(hspace)

    if thres is not None:
        hspace_thres[hspace_thres<=thres] = 0

    fix, axes = plt.subplots(figsize=(15, 8))
    axes.imshow(hspace_thres, cmap=plt.cm.bone,
                interpolation='nearest',
                aspect='auto')

    plt.draw()



def plot_hough(img, peak_angles, peak_dists, hspace, angles, dists):
    """
    This function plots the result
    :param img:
    :param peak_angles:
    :param peak_dists:
    :param angles:
    :return:
    """

    fix, axes = plt.subplots(1, 2, figsize=(15, 8))
    # =================================================
    # plot the original figure
    axes[0].imshow(1-img, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    axes[0].set_title('Input image')
    # plot the simple hough peak results
    axes[0].autoscale(False)

    # compute the peaks in cartesian
    for theta, dist in zip(peak_angles, peak_dists):
        x0 = dist/np.cos(-theta)
        x100 = 100.0*np.tan(-theta) + dist/np.cos((-theta))
        axes[0].plot([x0, x100], [0,100], linewidth=2, color='r')

    axes[1].imshow(
        hspace, cmap=plt.cm.bone,
        extent=(np.rad2deg(angles[-1]), np.rad2deg(angles[0]), dists[-1], dists[0]), aspect='auto', interpolation='nearest')
    # axes[1].scatter(-np.rad2deg(peak_angles), peak_dists, color='r', marker='*', s=100)

    axes[1].set_title('Hough transform')
    axes[1].set_xlabel('Angle (degree)')
    axes[1].set_ylabel('Distance (pixel)')

    plt.tight_layout()
    plt.draw()



if __name__ == "__main__":
    sys.exit(main(sys.argv))