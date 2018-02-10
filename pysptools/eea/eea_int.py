#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2014, Christian Therien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------
#
# eea_int.py - This file is part of the PySptools package.
#

"""
PPI, NFINDR, ATGP, FIPPI classes
"""


import numpy as np
from . import eea
from .inval import *
from .docstring import *


def _normalize(M):
    """
    Normalizes M to be in range [0, 1].

    Parameters:
      M: `numpy array`
          1D, 2D or 3D data.

    Returns: `numpy array`
          Normalized data.
    """
    minVal = np.min(M)
    maxVal = np.max(M)

    Mn = M - minVal;

    if maxVal == minVal:
        return np.zeros(M.shape);
    else:
        return Mn / (maxVal-minVal)


def _plot_end_members(path, E, utype, is_normalized, axes, suffix):
    """ Plot a endmembers graph using matplotlib """
    import os.path as osp
    import matplotlib.pyplot as plt
    if axes == None:
        axes = {}
        axes['wavelength'] = [x+1 for x in range(E.shape[1])]
        axes['x'] = 'Wavelength'
        axes['y'] = 'Brightness'
    else:
        if not('wavelength' in axes) or axes['wavelength'] == None:
            axes['wavelength'] = [x+1 for x in range(E.shape[1])]
        if not('x' in axes) or axes['x'] == None:
            axes['x'] = 'Wavelength'
        if not('y' in axes) or axes['y'] == None:
            axes['y'] = 'Brightness'

    plt.ioff()
    plt.xlabel(axes['x'])
    if is_normalized == True:
        plt.ylabel(axes['y']+' - normalized')
    else:
        plt.ylabel(axes['y'])
    plt.title('Spectral Profile')
    plt.grid(True)
    n_graph = 1
    legend = []
    for i in range(E.shape[0]):
        plt.plot(axes['wavelength'], E[i])
        legend.append('EM{0}'.format(str(i+1)))
        if (i+1) % 5 == 0 :
            plt.legend(legend, loc='upper left', framealpha=0.5)
            legend = []
            if suffix == None:
                fout = osp.join(path, 'emembers_{0}__{1}.png'.format(utype, n_graph))
            else:
                fout = osp.join(path, 'emembers_{0}__{1}_{2}.png'.format(utype, n_graph, suffix))
            try:
                plt.savefig(fout)
            except IOError:
                raise IOError('in _plot_end_members, no such file or directory: {0}'.format(path))
            n_graph += 1
            plt.clf()
            plt.xlabel(axes['x'])
            if is_normalized == True:
                plt.ylabel(axes['y']+' - normalized')
            else:
                plt.ylabel(axes['y'])
            plt.title('Spectral Profile')
            plt.grid(True)
    if E.shape[0] % 5 != 0:
        plt.legend(legend, loc='upper left', framealpha=0.5)
        if suffix == None:
            fout = osp.join(path, 'emembers_{0}__{1}.png'.format(utype, n_graph))
        else:
            fout = osp.join(path, 'emembers_{0}__{1}_{2}.png'.format(utype, n_graph, suffix))
        try:
            plt.savefig(fout)
        except IOError:
            raise IOError('in _plot_end_members, no such file or directory: {0}'.format(path))
    plt.close()


def _display_end_members(U, utype, is_normalized, axes, suffix):
    """ Display endmembers using matplotlib to the IPython Notebook. """
    import matplotlib.pyplot as plt
    if axes == None:
        axes = {}
        axes['wavelength'] = [x+1 for x in range(U.shape[1])]
        axes['x'] = 'Wavelength'
        axes['y'] = 'Brightness'
    else:
        if not('wavelength' in axes) or axes['wavelength'] == None:
            axes['wavelength'] = [x+1 for x in range(U.shape[1])]
        if not('x' in axes) or axes['x'] == None:
            axes['x'] = 'Wavelength'
        if not('y' in axes) or axes['y'] == None:
            axes['y'] = 'Brightness'

    plt.xlabel(axes['x'])
    if is_normalized == True:
        plt.ylabel(axes['y']+' - normalized')
    else:
        plt.ylabel(axes['y'])
    n_graph = 1
    legend = []
    for i in range(U.shape[0]):
        plt.plot(axes['wavelength'], U[i])
        legend.append('EM{0}'.format(str(i+1)))
        if (i+1) % 5 == 0 :
            plt.legend(legend, loc='upper left', framealpha=0.5)
            legend = []
            plt.xlabel(axes['x'])
            if is_normalized == True:
                plt.ylabel(axes['y']+' - normalized')
            else:
                plt.ylabel(axes['y'])
            if suffix == None:
                plt.title('Spectral Profile {0} - {1}'.format(n_graph, utype))
            else:
                plt.title('Spectral Profile {0} - {1} - {2}'.format(n_graph, utype, suffix))
            plt.grid(True)
            plt.show()
            plt.clf()
            n_graph += 1
    if U.shape[0] % 5 != 0:
        plt.legend(legend, loc='upper left', framealpha=0.5)
        plt.xlabel(axes['x'])
        if is_normalized == True:
            plt.ylabel(axes['y']+' - normalized')
        else:
            plt.ylabel(axes['y'])
        if suffix == None:
            plt.title('Spectral Profile {0} - {1}'.format(n_graph, utype))
        else:
            plt.title('Spectral Profile {0} - {1} - {2}'.format(n_graph, utype, suffix))
        plt.grid(True)
        plt.show()
    plt.close()


def _document(cls):
    import sys
    if sys.version_info[0] == 2:
        cls.plot.__func__.__doc__ = plot_docstring
        cls.display.__func__.__doc__ = display_docstring
    if sys.version_info[0] == 3:
        cls.plot.__doc__ = plot_docstring
        cls.display.__doc__ = display_docstring


def _compress(vec, mask):
    n = np.sum(mask)
    cmp = np.ndarray((n, vec.shape[1]), dtype=np.float)
    i = 0
    for j in range(mask.shape[0]):
        if mask[j] == 1:
            cmp[i] = vec[j]
            i += 1
    return cmp


class PPI(object):
    """
    Performs the pixel purity index algorithm for endmembers finding.
    """

    def __init__(self):
        self.E = None
        self.w = None
        self.idx = None
        self.idx3D = None
        self.is_normalized = False

    @ExtractInputValidation1('PPI')
    def extract(self, M, q, numSkewers=10000, normalize=False, mask=None):
        """
        Extract the endmembers.

        Parameters:
            M: `numpy array`
                A HSI cube (m x n x p).

            q: `int`
                Number of endmembers to find.

            numSkewers: `int [default 10000]`
                Number of "skewer" vectors to project data onto.
                In general, recommendation from the literature is 10000 skewers.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding signal is part of the
                endmembers search.

        Returns: `numpy array`
                Recovered endmembers (N x p).
        """
        if normalize == True:
            M = _normalize(M)
            self.is_normalized = True
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        self.q = q
        M = np.reshape(M, (self.w*h, M.shape[2]))

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (self.w*h))
            cM = _compress(M, m)
        else:
            cM = M

        self.E, self.idx = eea.PPI(cM, q, numSkewers)
        self.idx3D = [(i % self.w, i // self.w) for i in self.idx]
        return self.E

    def __str__(self):
        return 'pysptools.eea.eea_int.PPI object, hcube: {0}x{1}x{2}, n endmembers: {3}'.format(self.h, self.w, self.numBands, self.q)

    def get_idx(self):
        """
        Returns: `numpy array`
            Array of indices into the HSI cube corresponding to the
            induced endmembers
        """
        return self.idx3D

    @PlotInputValidation('PPI')
    def plot(self, path, axes=None, suffix=None):
        _plot_end_members(path, self.E, 'PPI', self.is_normalized, axes=axes, suffix=suffix)


    @DisplayInputValidation('PPI')
    def display(self, axes=None, suffix=None):
        _display_end_members(self.E, 'PPI', self.is_normalized, axes=axes, suffix=suffix)

_document(PPI)


class NFINDR(object):
    """
    N-FINDR endmembers induction algorithm.
    """

    def __init__(self):
        self.E = None
        self.Et = None
        self.w = None
        self.idx = None
        self.it = None
        self.idx3D = None
        self.is_normalized = False

    @ExtractInputValidation2('NFINDR')
    def extract(self, M, q, transform=None, maxit=None, normalize=False, ATGP_init=False, mask=None):
        """
        Extract the endmembers.

        Parameters:
            M: `numpy array`
                A HSI cube (m x n x p).

            q: `int`
                The number of endmembers to be induced.

            transform: `numpy array [default None]`
                The transformed 'M' cube by MNF (m x n x components). In this
                case the number of components must == q-1. If None, the built-in
                call to PCA is used to transform M in q-1 components.

            maxit: `int [default None]`
                The maximum number of iterations. Default is 3*p.

            normalize: `boolean [default False]`
                If True, M is normalized before doing the endmembers induction.

            ATGP_init: `boolean [default False]`
                Use ATGP to generate the first endmembers set instead
                of a random selection.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding signal is part of the
                endmembers search.

        Returns: `numpy array`
            Set of induced endmembers (N x p).

        References:
            Winter, M. E., "N-FINDR: an algorithm for fast autonomous spectral
            end-member determination in hyperspectral data", presented at the Imaging
            Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pgs. 266-275.

        Note:
            The division by (factorial(p-1)) is an invariant for this algorithm,
            for this reason it is skipped.
        """
        from . import nfindr
        if normalize == True:
            M = _normalize(M)
            self.is_normalized = True
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        self.q = q
        M = np.reshape(M, (self.w*h, M.shape[2]))
        if transform != None:
            transform = np.reshape(transform, (self.w*h, transform.shape[2]))

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (self.w*h))
            cM = _compress(M, m)
        else:
            cM = M
            
        self.E, self.Et, self.idx, self.it = nfindr.NFINDR(cM, q, transform, maxit, ATGP_init)
        self.idx3D = [(i % self.w, i // self.w) for i in self.idx]
        return self.E

    def __str__(self):
        return 'pysptools.eea.eea_int.NFINDR object, hcube: {0}x{1}x{2}, n endmembers: {3}'.format(self.h, self.w, self.numBands, self.q)

    def get_idx(self):
        """
        Returns : numpy array
            Array of indices into the HSI cube corresponding to the
            induced endmembers
        """
        return self.idx3D

    def get_iterations(self):
        """
        Returns : int
            The number of iterations.
        """
        return self.it

    def get_endmembers_transform(self):
        return self.Et

    @PlotInputValidation('NFINDR')
    def plot(self, path, axes=None, suffix=None):
        _plot_end_members(path, self.E, 'NFINDR', self.is_normalized, axes=axes, suffix=suffix)

    @DisplayInputValidation('NFINDR')
    def display(self, axes=None, suffix=None):
        _display_end_members(self.E, 'NFINDR', self.is_normalized, axes=axes, suffix=suffix)

_document(NFINDR)


class ATGP(object):
    """
    Automatic target generation process endmembers induction algorithm.
    """

    def __init__(self):
        self.E = None
        self.w = None
        self.idx = None
        self.idx3D = None
        self.is_normalized = False

    @ExtractInputValidation3('ATGP')
    def extract(self, M, q, normalize=False, mask=None):
        """
        Extract the endmembers.

        Parameters:
            M: `numpy array`
                A HSI cube (m x n x p).

            q: `int`
                Number of endmembers to be induced (positive integer > 0).

            normalize: `boolean [default False]`
                Normalize M before unmixing.

            mask: `numpy array [default None]`
                A binary mask, if True the corresponding signal is part of the
                endmembers search.

        Returns: `numpy array`
            Set of induced endmembers (N x p).

        References:
            A. Plaza y C.-I. Chang, "Impact of Initialization on Design of Endmember
            Extraction Algorithms", Geoscience and Remote Sensing, IEEE Transactions on,
            vol. 44, no. 11, pgs. 3397-3407, 2006.
        """
        if normalize == True:
            M = _normalize(M)
            self.is_normalized = True
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        self.q = q
        M = np.reshape(M, (self.w*h, M.shape[2]))

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (self.w*h))
            cM = _compress(M, m)
        else:
            cM = M

        self.E, self.idx = eea.ATGP(cM, q)
        self.idx3D = [(i % self.w, i // self.w) for i in self.idx]
        return self.E

    def __str__(self):
        return 'pysptools.eea.eea_int.ATGP object, hcube: {0}x{1}x{2}, n endmembers: {3}'.format(self.h, self.w, self.numBands, self.q)

    def get_idx(self):
        """
        Returns: `numpy array`
            Array of indices into the HSI cube corresponding to the
            induced endmembers
        """
        return self.idx3D

    @PlotInputValidation('ATGP')
    def plot(self, path, axes=None, suffix=None):
        _plot_end_members(path, self.E, 'ATGP', self.is_normalized, axes=axes, suffix=suffix)

    @DisplayInputValidation('ATGP')
    def display(self, axes=None, suffix=None):
        _display_end_members(self.E, 'ATGP', self.is_normalized, axes=axes, suffix=suffix)

_document(ATGP)


class FIPPI(object):
    """
    Fast Iterative Pixel Purity Index (FIPPI) endmembers
    induction algorithm.
    """

    def __init__(self):
        self.E = None
        self.w = None
        self.idx = None
        self.idx3D = None
        self.is_normalized = False

    @ExtractInputValidation4('FIPPI')
    def extract(self, M, q=None, maxit=None, normalize=False, mask=None):
        """
        Extract the endmembers.

        Parameters:
            M: `numpy array`
                A HSI cube (m x n x p).

            q: `int [default None]`
                Number of endmembers to be induced, if None use
                HfcVd to determine the number of endmembers to induce.

            maxit: `int [default None]`
                Maximum number of iterations. Default = 3*q.

            normalize: `boolean [default False]`
                Normalize M before unmixing.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding signal is part of the
                endmembers search.

        Returns: `numpy array`
            Set of induced endmembers (N x p).

        References:
            Chang, C.-I., "A fast iterative algorithm for implementation of pixel purity index",
            Geoscience and Remote Sensing Letters, IEEE, vol. 3, no. 1, pags. 63-67, 2006.
        """
        if normalize == True:
            M = _normalize(M)
            self.is_normalized = True
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        self.q = q
        M = np.reshape(M, (self.w*h, M.shape[2]))

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (self.w*h))
            cM = _compress(M, m)
        else:
            cM = M

        self.E, self.idx = eea.FIPPI(cM, q=q, maxit=maxit)
        self.idx3D = [(i % self.w, i // self.w) for i in self.idx]
        return self.E

    def __str__(self):
        return 'pysptools.eea.eea_int.FIPPI object, hcube: {0}x{1}x{2}, n endmembers: {3}'.format(self.h, self.w, self.numBands, self.q)

    def get_idx(self):
        """
        Returns: `numpy array`
            Array of indices into the HSI cube corresponding to the
            induced endmembers.
        """
        return self.idx3D

    @PlotInputValidation('FIPPI')
    def plot(self, path, axes=None, suffix=None):
        _plot_end_members(path, self.E, 'FIPPI', self.is_normalized, axes=axes, suffix=suffix)

    @DisplayInputValidation('FIPPI')
    def display(self, axes=None, suffix=None):
        _display_end_members(self.E, 'FIPPI', self.is_normalized, axes=axes, suffix=suffix)

_document(FIPPI)
