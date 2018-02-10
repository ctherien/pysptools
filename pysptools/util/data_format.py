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
# data_format.py - This file is part of the PySptools package.
#

"""
convert2d, convert3d, normalize functions
"""

from __future__ import division

import numpy as np


def convert2d(M):
    """
    Converts a 3D data cube (m x n x p) to a 2D matrix of points
    where N = m*n.

    Parameters:
        M: `numpy array`
            A HSI cube (m x n x p).

    Returns: `numpy array`
        2D data matrix (N x p)
    """

    if M.ndim != 3:
        raise RuntimeError('in formating.convert2d,  M have {0} dimension(s), expected 3 dimensions'.format(M.ndim))

    h, w, numBands = M.shape

    return np.reshape(M, (w*h, numBands))


def convert3d(M, h, w, sigLast=True):
    """
    Converts a 1D (N) or 2D matrix (p x N) or (N x p) to a 3D
    data cube (m x n x p) where N = m * n

    Parameters:
        N: `numpy array`
            1D (N) or 2D data matrix (p x N) or (N x p)

        h: `integer`
            Height axis length (or y axis) of the cube.

        w: `integer`
            Width axis length (or x axis) of the cube.

        siglast: `True [default False]`
            Determine if input N is (p x N) or (N x p).

    Returns: `numpy array`
            A 3D data cube (m x n x p)
    """

    if M.ndim > 2:
        raise RuntimeError('in formating.convert2d,  M have {0} dimension(s), expected 1 or 2 dimensions'.format(M.ndim))

    N = np.array(M)

    if sigLast == False:
        if N.ndim == 1:
            return np.reshape(N, (h, w), order='F')
        else:
            numBands, n = N.shape
            return np.reshape(N.transpose(), (h, w, numBands), order='F')

    if sigLast == True:
        if N.ndim == 1:
            return np.reshape(N, (h, w))
        else:
            numBands, n = N.shape
            return np.reshape(N.transpose(), (h, w, numBands), order='F')


def normalize(M):
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
