#
#
# These functions are usefull mainly for the development version, tests and demo.


#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function


import os.path as osp
import pysptools.util as util
import pysptools.skl as skl
from .cluster import Cluster


def get_scaled_img_and_class_map(spath, rpath, fname, rois, estimator, estimator_param, display=False):
    img_scaled = load_shrink_and_scale_img(spath, rpath, fname, display)
    img_ROIs = concat_ROIs(rpath, img_scaled, rois, fname, display)  
    cluster = Cluster(estimator=estimator, estimator_param=estimator_param)
    cmap = cluster.classify(img_scaled, img_ROIs)
    if display == True:
        if rpath == None:
            cluster.display(suffix=fname+' cluster')
        else:
            cluster.plot(rpath, suffix=fname+' cluster')
    return img_scaled, cmap


def plot_img(path, img, title):
    import matplotlib.pyplot as plt
    plt.ioff()
    fout = osp.join(path, title+'.png')
    plt.imshow(img)
    plt.savefig(fout)
    plt.close()


def display_img(img, title):
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.imshow(img, interpolation='none')
    if title == None:
        plt.title('No title')
    else:
        plt.title(title)
    plt.show()
    plt.close()


def _make_ROI(data, coords):
    r = util.ROIs(data.shape[0], data.shape[1])
    for coord in coords:
        r.add(*coord)
    return r


def load_reduce_and_scale_1(path, fname, n_shrink):
    M = util.load_mat_file(osp.join(path, fname))
    if n_shrink == 0:
        return skl.hyper_scale(M)
    m = M
    for n in range(n_shrink):
        m = skl.hyper_scale(util.shrink(m))
    return m

def load_reduce_and_scale(path, fname, n_shrink):
    M = util.load_mat_file(osp.join(path, fname))
    if n_shrink == 0:
        return skl.hyper_scale(M)
    m = M
    for n in range(n_shrink):
        m = util.shrink(m)
    return skl.hyper_scale(m)


def multi_shrink(M, n_shrink):
    if n_shrink == 0:
        return M
    m = M
    for n in range(n_shrink):
        m = util.shrink(m)
    return m


def batch_load(spath, samples, n_shrink):
    img_list = []
    for s in samples:
        img_list.append(load_reduce_and_scale(spath, s, n_shrink))
    return img_list


def load_shrink_and_scale_img(spath, rpath, fname, display=False):
    M = util.load_mat_file(osp.join(spath, fname))
    M = multi_shrink(M, 3)
    if display == True:
        if rpath == None:
            util.display_linear_stretch(M, 19, 13, 3, suffix=fname)
        else:
            util.plot_linear_stretch(M, rpath, 19, 13, 3, suffix=fname)
    return skl.hyper_scale(M)


def concat_ROIs(rpath, data, coords, label, display=False):
    r = util.ROIs(data.shape[0], data.shape[1])
    for coord in coords:
        r.add(*coord)
    if display == True:
        if rpath == None:
            r.display(colorMap='Paired', suffix=' '+label+' cluster')
        else:
            r.plot(rpath, colorMap='Paired', suffix=' '+label+' cluster')
    return r


def batch_classify(spath, rpath, model, samples, n_shrink):
    for s in samples:
        M = util.load_mat_file(osp.join(spath, s))   
        M = multi_shrink(M, n_shrink)
        M_scaled = skl.hyper_scale(M)
        if rpath == None:
            util.display_linear_stretch(M, 19, 13, 3, suffix=s)
        else:
            util.plot_linear_stretch(M, rpath, 19, 13, 3, suffix=s)
        model.classify(M_scaled)
        if rpath == None:
            model.display(suffix='batch '+s)
        else:
            model.plot(rpath, suffix='batch '+s)
