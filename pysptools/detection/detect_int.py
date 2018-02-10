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
# detect_int.py - This file is part of the PySptools package.
#


"""
MatchedFilter, ACE, CEM, GLRT, OSP classes
"""


import numpy as np
from . import detect
from .inval import *
from .docstring import *


def _plot_target_map(path, tmap, map_type, whiteOnBlack, suffix=None):
    """ Plot a target map using matplotlib """
    import matplotlib.pyplot as plt
    import os.path as osp
    if path != None:
        plt.ioff()
    img = plt.imshow(tmap)
    if whiteOnBlack == True:
        img.set_cmap('Greys_r')
    elif whiteOnBlack == False:
        img.set_cmap('Greys')
    else:
        # throw an error?
        img.set_cmap('Blues')
    if path != None:
        if suffix == None:
            fout = osp.join(path, 'tmap_{0}.png'.format(map_type))
        else:
            fout = osp.join(path, 'tmap_{0}_{1}.png'.format(map_type, suffix))
        try:
            plt.savefig(fout)
        except IOError:
            raise IOError('in detection._plot_target_map, no such file or directory: {0}'.format(path))
    else:
        if suffix == None:
            plt.title('{0} Target Map'.format(map_type))
        else:
            plt.title('{0} Target Map - {1}'.format(map_type, suffix))
            plt.show()
    plt.clf()


def _document(cls):
    import sys
    if sys.version_info[0] == 2:
        cls.plot.__func__.__doc__ = plot_docstring
        cls.display.__func__.__doc__ = display_docstring
    if sys.version_info[0] == 3:
        cls.plot.__doc__ = plot_docstring
        cls.display.__doc__ = display_docstring


class MatchedFilter(object):
    """
    Performs the matched filter algorithm for target detection.
    """

    def __init__(self):
        self.target_map = None

    @DetectInputValidation1('MatchedFilter')
    def detect(self, M, t, threshold=None):
        """
        Parameters:
          M: `numpy array`
            A HSI cube (m x n x p).

          t: `numpy array`
            A target pixel (p).

          threshold: `float or None [default None]`
            Apply a threshold to the detection result.
            Usefull to isolate the result.

        Returns: `numpy array`
            Vector of detector output (m x n x 1).

        References:
            Qian Du, Hsuan Ren, and Chein-I Cheng. A Comparative Study of
            Orthogonal Subspace Projection and Constrained Energy Minimization.
            IEEE TGRS. Volume 41. Number 6. June 2003.
        """
        h,w,numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        target = detect.MatchedFilter(Mr, t)
        self.target_map = np.reshape(target, (h, w))
        if threshold != None:
            self.target_map = self.target_map > threshold
        return self.target_map

    def __str__(self):
        return 'pysptools.detection.detect_int.MatchedFilter object, hcube: {0}x{1}x{2}'.format(self.h, self.w, self.numBands)

    @PlotInputValidation('MatchedFilter')
    def plot(self, path, whiteOnBlack=True, suffix=None):
        _plot_target_map(path, self.target_map, 'MatchedFilter', whiteOnBlack, suffix)

    @DisplayInputValidation('MatchedFilter')
    def display(self, whiteOnBlack=True, suffix=None):
        _plot_target_map(None, self.target_map, 'MatchedFilter', whiteOnBlack, suffix)

_document(MatchedFilter)


class ACE(object):
    """
    Performs the adaptive cosin/coherent estimator algorithm for target
    detection.
    """

    def __init__(self):
        self.target_map = None

    @DetectInputValidation1('ACE')
    def detect(self, M, t, threshold=None):
        """
        Parameters:
          M: `numpy array`
            A HSI cube (m x n x p).

          t: `numpy array`
            A target pixel (p).

          threshold: `float or None [default None]`
            Apply a threshold to the detection result.
            Usefull to isolate the result.

        Returns: `numpy array`
            Vector of detector output (m x n x 1).

        References:
          X Jin, S Paswater, H Cline.  "A Comparative Study of Target Detection
          Algorithms for Hyperspectral Imagery."  SPIE Algorithms and Technologies
          for Multispectral, Hyperspectral, and Ultraspectral Imagery XV.  Vol
          7334.  2009.
        """
        h,w,numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        target = detect.ACE(Mr, t)
        self.target_map = np.reshape(target, (h, w))
        if threshold != None:
            self.target_map = self.target_map > threshold
        return self.target_map

    def __str__(self):
        return 'pysptools.detection.detect_int.ACE object, hcube: {0}x{1}x{2}'.format(self.h, self.w, self.numBands)

    @PlotInputValidation('ACE')
    def plot(self, path, whiteOnBlack=True, suffix=None):
        _plot_target_map(path, self.target_map, 'ACE', whiteOnBlack, suffix)

    @DisplayInputValidation('ACE')
    def display(self, whiteOnBlack=True, suffix=None):
        _plot_target_map(None, self.target_map, 'ACE', whiteOnBlack, suffix)

_document(ACE)


class CEM(object):
    """
    Performs the constrained energy minimization algorithm for target
    detection.
    """

    def __init__(self):
        self.target_map = None

    @DetectInputValidation1('CEM')
    def detect(self, M, t, threshold=None):
        """
        Parameters:
          M: `numpy array`
            A HSI cube (m x n x p).

          t: `numpy array`
            A target pixel (p).

          threshold: `float or None [default None]`
            Apply a threshold to the detection result.
            Usefull to isolate the result.

        Returns: `numpy array`
            Vector of detector output (m x n x 1).

        References:
            Qian Du, Hsuan Ren, and Chein-I Cheng. A Comparative Study of
            Orthogonal Subspace Projection and Constrained Energy Minimization.
            IEEE TGRS. Volume 41. Number 6. June 2003.
        """
        h,w,numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        target = detect.CEM(Mr, t)
        self.target_map = np.reshape(target, (h, w))
        if threshold != None:
            self.target_map = self.target_map > threshold
        return self.target_map

    def __str__(self):
        return 'pysptools.detection.detect_int.CEM object, hcube: {0}x{1}x{2}'.format(self.h, self.w, self.numBands)

    @PlotInputValidation('CEM')
    def plot(self, path, whiteOnBlack=True, suffix=None):
        _plot_target_map(path, self.target_map, 'CEM', whiteOnBlack, suffix)

    @DisplayInputValidation('CEM')
    def display(self, whiteOnBlack=True, suffix=None):
        _plot_target_map(None, self.target_map, 'CEM', whiteOnBlack, suffix)

_document(CEM)


class GLRT(object):
    """
    Performs the generalized likelihood test ratio algorithm for target
    detection.
    """

    def __init__(self):
        self.target_map = None

    @DetectInputValidation1('GLRT')
    def detect(self, M, t, threshold=None):
        """
        Parameters:
          M: `numpy array`
            A HSI cube (m x n x p).

          t: `numpy array`
            A target pixel (p).

          threshold: `float or None [default None]`
            Apply a threshold to the detection result.
            Usefull to isolate the result.

        Returns: `numpy array`
            Vector of detector output (m x n x 1).

        References
            T. F. AyouB, "Modified GLRT Signal Detection Algorithm," IEEE
            Transactions on Aerospace and Electronic Systems, Vol 36, No 3, July
            2000.
        """
        h,w,numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        target = detect.GLRT(Mr, t)
        self.target_map = np.reshape(target, (h, w))
        if threshold != None:
            self.target_map = self.target_map > threshold
        return self.target_map

    def __str__(self):
        return 'pysptools.detection.detect_int.GLRT object, hcube: {0}x{1}x{2}'.format(self.h, self.w, self.numBands)

    @PlotInputValidation('GLRT')
    def plot(self, path, whiteOnBlack=True, suffix=None):
        _plot_target_map(path, self.target_map, 'GLRT', whiteOnBlack, suffix)

    @DisplayInputValidation('GLRT')
    def display(self, whiteOnBlack=True, suffix=None):
        _plot_target_map(None, self.target_map, 'GLRT', whiteOnBlack, suffix)

_document(GLRT)


class OSP(object):
    """
    Performs the othogonal subspace projection algorithm for target
    detection.
    """

    def __init__(self):
        self.target_map = None

    @DetectInputValidation2('OSP')
    def detect(self, M, E, t, threshold=None):
        """
        Parameters:
          M: `numpy array`
            A HSI cube (m x n x p).

          E: `numpy array`
            Background pixels (n x p).

          t: `numpy array`
            A target pixel (p).

          threshold: `float or None [default None]`
            Apply a threshold to the detection result.
            Usefull to isolate the result.

        Returns: `numpy array`
            Vector of detector output (m x n x 1).

        References:
            Qian Du, Hsuan Ren, and Chein-I Cheng. "A Comparative Study of
            Orthogonal Subspace Projection and Constrained Energy Minimization."
            IEEE TGRS. Volume 41. Number 6. June 2003.
        """
        h,w,numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        target = detect.OSP(Mr, E, t)
        self.target_map = np.reshape(target, (h, w))
        if threshold != None:
            self.target_map = self.target_map > threshold
        return self.target_map

    def __str__(self):
        return 'pysptools.detection.detect_int.OSP object, hcube: {0}x{1}x{2}'.format(self.h, self.w, self.numBands)

    @PlotInputValidation('OSP')
    def plot(self, path, whiteOnBlack=True, suffix=None):
        _plot_target_map(path, self.target_map, 'OSP', whiteOnBlack, suffix)

    @DisplayInputValidation('OSP')
    def display(self, whiteOnBlack=True, suffix=None):
        _plot_target_map(None, self.target_map, 'OSP', whiteOnBlack, suffix)

_document(OSP)
