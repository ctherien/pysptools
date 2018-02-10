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
# dist.py - This file is part of the PySptools package.
#

"""
SAM, SID, NormXCorr, chebyshev functions
"""

from __future__ import division

import numpy as np
import math


def SAM(s1, s2):
    """
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            The angle between vectors s1 and s2 in radians.
    """
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        # python math don't like when acos is called with
        # a value very near to 1
        return 0.0
    return angle


def SID(s1, s2):
    """
    Computes the spectral information divergence between two vectors.

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            Spectral information divergence between s1 and s2.

    Reference
        C.-I. Chang, "An Information-Theoretic Approach to SpectralVariability,
        Similarity, and Discrimination for Hyperspectral Image"
        IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 46, NO. 5, AUGUST 2000.

    """
    p = (s1 / np.sum(s1)) + np.spacing(1)
    q = (s2 / np.sum(s2)) + np.spacing(1)
    return np.sum(p * np.log(p / q) + q * np.log(q / p))


def chebyshev(s1, s2):
    """
    Computes the chebychev distance between two vector.

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            Chebychev distance between s1 and s2.
    """
    return np.amax(np.abs(s1 - s2))


def NormXCorr(s1, s2):
    """
    Computes the normalized cross correlation distance between two vector.

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            NormXCorr distance between s1 and s2, dist is between [-1, 1].
            A value of one indicate a perfect match.
    """
    # s1 and s2 have the same length
    import scipy.stats as ss
    s = s1.shape[0]
    corr = np.sum((s1 - np.mean(s1)) * (s2 - np.mean(s2))) / (ss.tstd(s1) * ss.tstd(s2))
    return corr * (1./(s-1))


def classify(fn, M, E):
    """
    Classify SAM or SID on a HSI cube
    Can't be use with NormXCorr
    """
    import pysptools.util as util
    width, height, bands = M.shape
    M = util.convert2d(M)
    cmap = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        T = M[i]
        floor = np.PINF
        k = 0
        for j in range(E.shape[0]):
            R = E[j]
            result = fn(T, R)
            if result < floor:
                floor = result
                k = j
        cmap[i] = k
    return util.convert3d(cmap, width, height)
