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
# amaps_int.py - This file is part of the PySptools package.
#

"""
UCLS, NNLS, FCLS classes
"""

import os.path as osp
import numpy as np
from . import amaps
from .inval import *


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


def _plot_abundance_map(amap, path, map_type, mask, interpolation, colorMap, columns, suffix):
    """ Plot an abundance map using matplotlib """

    def one_figure(amap, path, map_type, mask, interpolation, colorMap, columns, suffix):
        rows, remainder = divmod(amap.shape[2], columns)
        if remainder != 0: rows += 1
        n_amaps = amap.shape[2]
        f, axes = plt.subplots(rows, columns, figsize=(4*columns, 4*rows))
        f.set_dpi(128)
        #f.set_dpi(64)
        if suffix == None:
            f.suptitle('{0} Inversion'.format(map_type), fontsize=12)
        else:
            f.suptitle('{0} Inversion {1}'.format(map_type, suffix), fontsize=12)
        i = 1
        for r in range(rows):
            for c in range(columns):
                axes[r,c].axis('off')
        for r in range(rows):
            for c in range(columns):
                if i > n_amaps: continue
                m = amap[:,:,i-1]
                if isinstance(mask, np.ndarray):
                    m = m * mask
                ax = axes[r,c].imshow(m, interpolation=interpolation)
                ax.set_cmap(colorMap)
                axes[r,c].set_title('EM{}'.format(i), fontsize=10)
                i += 1
        if path != None:
            if suffix == None:
                fout = osp.join(path, '{0}.png'.format(map_type))
            else:
                fout = osp.join(path, '{0}_{1}.png'.format(map_type, suffix))
            try:
#                f.savefig(fout, dpi='figure')
                f.savefig(fout)
            except IOError:
                raise IOError('in abundance_map._plot_abundance_map, no such file or directory: {0}'.format(path))
        else:
            plt.show()
        plt.clf()

    def multi_figures(amap, path, map_type, mask, interpolation, colorMap, suffix):
        for i in range(amap.shape[2]):
            m = amap[:,:,i]

            if isinstance(mask, np.ndarray):
                m = m * mask

            img = plt.imshow(m, interpolation=interpolation)
            img.set_cmap(colorMap)
            plt.colorbar()
            if path != None:
                if suffix == None:
                    fout = osp.join(path, '{0}_{1}.png'.format(map_type, i+1))
                else:
                    fout = osp.join(path, '{0}_{1}_{2}.png'.format(map_type, i+1, suffix))
                try:
                    plt.savefig(fout)
                except IOError:
                    raise IOError('in abundance_map._plot_abundance_map, no such file or directory: {0}'.format(path))
            else:
                if suffix == None:
                    plt.title('{0} Inversion - EM{1}'.format(map_type, i+1))
                else:
                    plt.title('{0} Inversion - EM{1} - {2}'.format(map_type, i+1, suffix))
                plt.show()
            plt.clf()

    import matplotlib.pyplot as plt
    import warnings
    if path != None:
        plt.ioff()
    if columns != None and columns >= amap.shape[2]:
         warnings.warn('In abundance_map._plot_abundance_map, the number of abundances map to display is less or equal the number of columns')
    if columns != None and columns < amap.shape[2]:
        one_figure(amap, path, map_type, mask, interpolation, colorMap, columns, suffix)
    else:
        multi_figures(amap, path, map_type, mask, interpolation, colorMap, suffix)


_plot_docstring = """
        Plot the abundance maps.

        Parameters:
            path: `string`
              The path where to put the plot.

            mask: `numpy array [default None]`
              A binary mask, when *True* the selected pixel is displayed.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default jet]`
              A matplotlib color map.

            columns: `int [default None]`
              Display all the images in one figure organized by
              columns.

            suffix: `string [default None]`
              Suffix to add to the file name.
        """


_display_docstring = """
        Display the abundance maps to a IPython Notebook.

        Parameters:
            mask: `numpy array [default None]`
                A binary mask, when *True* the selected pixel is displayed.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default jet]`
              A matplotlib color map.

            columns: `int [default None]`
              Display all the images in one figure organized by
              columns.

            suffix: `string [default None]`
              Suffix to add to the title.
        """


def _document(cls):
    import sys
    if sys.version_info[0] == 2:
        cls.plot.__func__.__doc__ = _plot_docstring
        cls.display.__func__.__doc__ = _display_docstring
    if sys.version_info[0] == 3:
        cls.plot.__doc__ = _plot_docstring
        cls.display.__doc__ = _display_docstring


def _compress(vec, mask):
    n = np.sum(mask)
    cmp = np.ndarray((n, vec.shape[1]), dtype=np.float)
    i = 0
    for j in range(mask.shape[0]):
        if mask[j] == 1:
            cmp[i] = vec[j]
            i += 1
    return cmp


def _expand(amap, mask, l, q):
    exp = np.zeros((l,q), dtype=np.float)
    i = 0
    for j in range(mask.shape[0]):
        if mask[j] == 1:
            exp[j] = amap[i]
            i += 1
    return exp


class UCLS(object):
    """
    Performs unconstrained least squares abundance estimation.
    """

    def __init__(self):
        self.amap = None
        self.m = None
        self.n = None
        self.q = None

    @MapInputValidation('UCLS')
    def map(self, M, U, normalize=False, mask=None):
        """
        Performs unconstrained least squares abundance estimation on
        the HSI cube M using the spectral library U.

        Parameters:
       	M: `numpy array`
             A HSI cube (m x n x p).

          U: `numpy array`
             A spectral library of endmembers (q x p).

          normalize: `boolean [default False]`
             If True, M and U are normalized before doing the spectra mapping.

          mask: `numpy array [default None]`
             A binary mask, when *True* the selected pixel is unmixed.

        Returns: `numpy array`
              An abundance maps (m x n x q).
        """
        h,w,numBands = M.shape
        if normalize == True:
            M = _normalize(M)
            U = _normalize(U)
        Mr = np.reshape(M, (w*h, numBands))
        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (w*h))
            cMr = _compress(Mr, m)
            c_amap2D = amaps.UCLS(cMr, U)
            amap2D = _expand(c_amap2D, m, w*h, U.shape[0])
        else:
            amap2D = amaps.UCLS(Mr, U)
        self.amap = np.reshape(amap2D, (h, w, U.shape[0]))
        self.m = h
        self.n = w
        self.q = U.shape[0]
        return self.amap

    def __str__(self):
        return 'pysptools.abundance_maps.amaps_int.UCLS object, maps: {0}x{1}x{2}'.format(self.m, self.n, self.q)

    @PlotInputValidation('UCLS')
    def plot(self, path, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, path, 'UCLS', mask, interpolation, colorMap, columns, suffix)

    @DisplayInputValidation('UCLS')
    def display(self, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, None, 'UCLS', mask, interpolation, colorMap, columns, suffix)

_document(UCLS)


class NNLS(object):
    """
    NNLS performs non-negative constrained least
    squares with the abundance nonnegative constraint (ANC).
    Utilizes the method of Bro.
    """

    def __init__(self):
        self.amap = None
        self.m = None
        self.n = None
        self.q = None

    @MapInputValidation('NNLS')
    def map(self, M, U, normalize=False, mask=None):
        """
        NNLS performs non-negative constrained least squares of each pixel
        in M using the endmember signatures of U.

        Parameters:
       	M: `numpy array`
             A HSI cube (m x n x p).

          U: `numpy array`
             A spectral library of endmembers (q x p).

          normalize: `boolean [default False]`
             If True, M and U are normalized before doing the spectra mapping.

          mask: `numpy array [default None]`
             A binary mask, when *True* the selected pixel is unmixed.

        Returns: `numpy array`
              An abundance maps (m x n x q).
        """
        h,w,numBands = M.shape
        if normalize == True:
            M = _normalize(M)
            U = _normalize(U)
        Mr = np.reshape(M, (w*h, numBands))
        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (w*h))
            cMr = _compress(Mr, m)
            c_amap2D = amaps.NNLS(cMr, U)
            amap2D = _expand(c_amap2D, m, w*h, U.shape[0])
        else:
            amap2D = amaps.NNLS(Mr, U)
        self.amap = np.reshape(amap2D, (h, w, U.shape[0]))
        self.m = h
        self.n = w
        self.q = U.shape[0]
        return self.amap

    def __str__(self):
        return 'pysptools.abundance_maps.amaps_int.NNLS object, maps: {0}x{1}x{2}'.format(self.m, self.n, self.q)

    @PlotInputValidation('NNLS')
    def plot(self, path, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, path, 'NNLS', mask, interpolation, colorMap, columns, suffix)

    @DisplayInputValidation('NNLS')
    def display(self, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, None, 'NNLS', mask, interpolation, colorMap, columns, suffix)

_document(NNLS)


class FCLS(object):
    """
    Performs fully constrained least squares. Fully constrained least squares
    is least squares with the abundance sum-to-one constraint (ASC) and the
    abundance nonnegative constraint (ANC).
    """

    def __init__(self):
        self.amap = None
        self.m = None
        self.n = None
        self.q = None

    @MapInputValidation('FCLS')
    def map(self, M, U, normalize=False, mask=None):
        """
        Performs fully constrained least squares of each pixel in M
        using the endmember signatures of U.

        Parameters:
       	M: `numpy array`
             A HSI cube (m x n x p).

          U: `numpy array`
             A spectral library of endmembers (q x p).

          normalize: `boolean [default False]`
             If True, M and U are normalized before doing the spectra mapping.

          mask: `numpy array [default None]`
             A binary mask, when *True* the selected pixel is unmixed.

        Returns: `numpy array`
              An abundance maps (m x n x q).
        """
        h,w,numBands = M.shape
        if normalize == True:
            M = _normalize(M)
            U = _normalize(U)
        Mr = np.reshape(M, (w*h, numBands))
        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (w*h))
            cMr = _compress(Mr, m)
            c_amap2D = amaps.FCLS(cMr, U)
            amap2D = _expand(c_amap2D, m, w*h, U.shape[0])
        else:
            amap2D = amaps.FCLS(Mr, U)
        self.amap = np.reshape(amap2D, (h, w, U.shape[0]))
        self.m = h
        self.n = w
        self.q = U.shape[0]
        return self.amap

    def __str__(self):
        return 'pysptools.abundance_maps.amaps_int.FCLS object, maps: {0}x{1}x{2}'.format(self.m, self.n, self.q)

    @PlotInputValidation('FCLS')
    def plot(self, path, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, path, 'FCLS', mask, interpolation, colorMap, columns, suffix)

    @DisplayInputValidation('FCLS')
    def display(self, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, None, 'FCLS', mask, interpolation, colorMap, columns, suffix)

_document(FCLS)
