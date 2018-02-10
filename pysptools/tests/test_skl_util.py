# Tested on Python 3.5 only

import numpy as np
import pysptools.util as util
import pickle

import pysptools.eea as eea
import pysptools.abundance_maps as amp


def remove_bands(M):
    """
    Remove the bands with atmospheric
    scattering.
    Remove:
        [0..4]
        [102..110]
        [148..169]
        [211..end]
    """
    p1 = list(range(5,102))
    p2 = list(range(111,148))
    p3 = list(range(170,211))
    Mp = M[:,:,p1+p2+p3]
    # Resample the wavelength axis
    wavelength = range(1, Mp.shape[2]+1)
    return Mp, wavelength


class DataMine(object):

    def __init__(self, hcube, n_em, suffix):
        self.suffix = suffix
        self.nfindr = eea.NFINDR()
        self.U = self.nfindr.extract(hcube, n_em, maxit=5, normalize=False, ATGP_init=True)
#        self.xxls = amp.FCLS()
        self.xxls = amp.NNLS()
        self.amaps = self.xxls.map(hcube, self.U, normalize=False)
        
    def get_endmembers(self):
        return self.U

    def get_abundances(self):
        return self.amaps

    def plot_endmembers(self, path, axes=None):
        self.nfindr.plot(path, axes=axes, suffix=self.suffix)

    def plot_abundances(self, path):
        self.xxls.plot(path, colorMap='gist_earth', columns=3, suffix=self.suffix)

    def display_endmembers(self, axes=None):
        self.nfindr.display(axes=axes, suffix=self.suffix)

    def display_abundances(self):
        self.xxls.display(colorMap='gist_earth', columns=3, suffix=self.suffix)


class Mask(object):
    mask = None
    
    def __init__(self, label):
        self.label = label
        
    def put1(self, M, mp1, t1):
        self.mask = mp1 > t1
        self.roi = util.ROIs(M.shape[0], M.shape[1])
        self.roi.add(self.label, {'raw': self.mask})

    def put2(self, M, mp1, t1, mp2, t2):
        self.mask = np.logical_or((mp1 > t1), (mp2 > t2))
        self.roi = util.ROIs(M.shape[0], M.shape[1])
        self.roi.add(self.label, {'raw': self.mask})

    def get_mask(self):
        return self.mask

    def get_roi(self):
        return self.roi

    def dump(self, fname):
        pickle.dump(self.mask, open(fname, "wb" ))
        
    def load(self, fname):
        self.mask = pickle.load(open(fname, "rb" ))
        
    def plot(self, path, suffix=''):
        self.roi.plot(path, suffix=suffix)

    def display(self, colorMap='jet', suffix=''):
        self.roi.display(colorMap='Paired', suffix=suffix)


class HyperEstimatorTraining(object):
    """ 
    This is a training example. If you set 'split' to 0.33, you train on
    one third of the cube and classify on all the cube.
    """

    def __init__(self, estimator, hcube, mask, split, label, **kwargs):
        self.estimator = estimator
        self.label = label
        train_cube, train_mask = self._split(hcube, mask, split)
        self.model = self._train_section(train_cube, train_mask, kwargs, label)
        self.model.classify(hcube)
        
    def _train_section(self, cube_section, mask_section, params, feature_name):
        roi = util.ROIs(cube_section.shape[0], cube_section.shape[1])
        roi.add(feature_name, {'raw': mask_section})
        model = self.estimator(**params)
        model.fit_rois(cube_section, roi)
        return model
        
    def _split(self, hcube, mask, frac):
        y = hcube.shape[1]
        v = int(y * frac)
        X3d = hcube[:,0:v,:]
        Y3d = mask[:,0:v]
        return X3d, Y3d

    def _convert2D(self, M):
        h, w, numBands = M.shape
        return np.reshape(M, (w*h, numBands))

    def get_model(self):
        return self.model
