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
# nfindr.py - This file is part of the PySptools package.
#

"""
NFINDR function
"""

import math
import random
import numpy as np
import scipy as sp
from . import eea


def NFINDR(data, q, transform=None, maxit=None, ATGP_init=False):
    """
    N-FINDR endmembers induction algorithm.

    Parameters:
        data: `numpy array`
            Column data matrix [nvariables x nsamples].

        q: `int`
            Number of endmembers to be induced.

        transform: `numpy array [default None]`
            The transformed 'data' matrix by MNF (N x components). In this
            case the number of components must == q-1. If None, the built-in
            call to PCA is used to transform the data.

        maxit: `int [default None]`
            Maximum number of iterations. Default = 3*q.

        ATGP_init: `boolean [default False]`
            Use ATGP to generate the first endmembers set instead
            of a random selection.

    Returns: `tuple: numpy array, numpy array, int`
        * Set of induced endmembers (N x p)
        * Set of transformed induced endmembers (N x p)
        * Array of indices into the array data corresponding to the
          induced endmembers
        * The number of iterations.

    References:
      Winter, M. E., "N-FINDR: an algorithm for fast autonomous spectral
      end-member determination in hyperspectral data", presented at the Imaging
      Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pgs. 266-275.
    """
    # data size
    nsamples, nvariables = data.shape

    if maxit == None:
        maxit = 3*q

    if transform == None:
        # transform as shape (N x p)
        transform = data
        transform = eea._PCA_transform(data, q-1)
    else:
        transform = transform

    # Initialization
    # TestMatrix is a square matrix, the first row is set to 1
    TestMatrix = np.zeros((q, q), dtype=np.float32, order='F')
    TestMatrix[0,:] = 1
    IDX = None
    if ATGP_init == True:
        induced_em, idx = eea.ATGP(transform, q)
        IDX = np.array(idx, dtype=np.int64)
        for i in range(q):
            TestMatrix[1:q, i] = induced_em[i]
    else:
        IDX =  np.zeros((q), dtype=np.int64)
        for i in range(q):
            idx = int(math.floor(random.random()*nsamples))
            TestMatrix[1:q, i] = transform[idx]
            IDX[i] = idx

    actualVolume = 0
    it = 0
    v1 = -1.0
    v2 = actualVolume

    while it <= maxit and v2 > v1:
        for k in range(q):
            for i in range(nsamples):
                TestMatrix[1:q, k] = transform[i]
                volume = math.fabs(sp.linalg._flinalg.sdet_c(TestMatrix)[0])
                if volume > actualVolume:
                    actualVolume = volume
                    IDX[k] = i
            TestMatrix[1:q, k] = transform[IDX[k]]
        it = it + 1
        v1 = v2
        v2 = actualVolume

    E = np.zeros((len(IDX), nvariables), dtype=np.float32)
    Et = np.zeros((len(IDX), q-1), dtype=np.float32)
    for j in range(len(IDX)):
        E[j] = data[IDX[j]]
        Et[j] = transform[IDX[j]]

    return E, Et, IDX, it
