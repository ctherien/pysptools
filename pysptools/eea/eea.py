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
# eea.py - This file is part of the PySptools package.
#

"""
ATGP, FIPPI, PPI functions
"""

from __future__ import print_function

import numpy as np
import scipy as sp


def _PCA_transform(M, n_components):
    from sklearn.decomposition import PCA
    # M as shape (N x p)
    # scikit.learn expect (N x p)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(M)


# not a efficient way of doing this
def _rows_union(a, b):
    a_hashable = map(tuple, a)
    b_hashable = map(tuple, b)
    a_set = set(a_hashable)
    b_set = set(b_hashable)
    if a_set.issubset(b_set):
        # return True if a is subset of b, False otherwise
        return b, True
    u = a_set.union(b_set)
    return np.array(list(u)), False


def ATGP(data, q):
    """
    Automatic Target Generation Process endmembers induction algorithm

    Parameters:
        data: `numpy array`
            2d matrix of HSI data ((m x n) x p)

        q: `int`
            Number of endmembers to be induced (positive integer > 0)

    Returns: `tuple: numpy array, numpy array`
        * Set of induced endmembers (N x p).
        * Induced endmembers indexes vector.

    References:
      A. Plaza, C.-I. Chang, "Impact of Initialization on Design of Endmember
      Extraction Algorithms", Geoscience and Remote Sensing, IEEE Transactions on,
      vol. 44, no. 11, pgs. 3397-3407, 2006.
    """
    nsamples, nvariables = data.shape

    # Algorithm initialization
    # the sample with max energy is selected as the initial endmember
    max_energy = -1
    idx = 0
    for i in range(nsamples):
        r = data[i]
        val = np.dot(r, r)
        if val > max_energy:
          max_energy = val
          idx = i

    # Initialization of the set of endmembers and the endmembers index vector
    E = np.zeros((q, nvariables), dtype=np.float32)
    E[0] = data[idx] # the first endmember selected
    # Generate the identity matrix.
    I = np.eye(nvariables)
    IDX = np.zeros(q, dtype=np.int)

    IDX[0] = idx

    for i in range(q-1):
        UC = E[0:i+1]
        # Calculate the orthogonal projection with respect to the pixels at present chosen.
        # This part can be replaced with any other distance
        PU = I - np.dot(UC.T,np.dot(sp.linalg.pinv(np.dot(UC,UC.T)),UC))
        max_energy = -1
        idx = 0
        # Calculate the most different pixel from the already selected ones according to
        # the orthogonal projection (or any other distance selected)
        for j in range(nsamples):
            r = data[j]
            result = np.dot(PU, r)
            val = np.dot(result.T, result)
            if val > max_energy:
                max_energy = val
                idx = j
    # The next chosen pixel is the most different from the already chosen ones
        E[i+1] = data[idx]
        IDX[i+1] = idx

    return E, IDX


def FIPPI(M, q=None, far=None, maxit=None):
    """
    Fast Iterative Pixel Purity Index (FIPPI) endmqbers
    induction algorithm.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data ((m x n) x p)

        q: `int [default None]`
            Number of endmembers to be induced, if None use
            HfcVd to determine the number of endmembers to induce.

        far: `float [default None]`
            False alarm rate(s), a parameter to HfcVd equal to 10**-5
            by default.

        maxit: `int [default None]
            Maximum number of iterations. Default = 3*p.

    Returns: `tuple: numpy array, numpy array`
        * Set of induced endmembers (N x p).
        * Array of indices into the array data corresponding to the
          induced endmembers.

    References:
        Chang, C.-I., "A fast iterative algorithm for implementation of pixel purity index",
        Geoscience and Remote Sensing Letters, IEEE, vol. 3, no. 1, pags. 63-67, 2006.
    """
    import pysptools.material_count.vd as vd
    N, p1 = M.shape

    if q == None:
        if far == None:
            far = 10**-5
        q = vd.HfcVd(M, far=far)[0]
        print("In FIPPI, virtual dimensionality is:", q)
    if maxit == None:
        maxit = 0

    data_pca = _PCA_transform(M, q)
    N, p = data_pca.shape

    # Initial skewers
    skewers = ATGP(data_pca, q)[0]

    stop = False # stop condition
    it = 1 # iterations
    idx = [] # indexes of the induced endmembers
    while not stop:
        # Calculate Nppi
        Nppi = np.zeros((N), dtype=np.int32)
        proj = np.dot(data_pca, skewers.T)
        I1 = np.argmin(proj, axis=0)
        I2 = np.argmax(proj, axis=0)
        for j in range(proj.shape[1]):
            Nppi[I1[j]] = Nppi[I1[j]] + 1
            Nppi[I2[j]] = Nppi[I2[j]] + 1
        # Check new skewers
        # A tuple is returned, first part is the array r
        r = np.nonzero(Nppi)[0]
        skewers_r, isSubset = _rows_union(data_pca[r], skewers)
        # if data_pca[r] isSubset of skewers
        if isSubset == True:
            stop = True
            idx = r
        else:
            # new skewers
            skewers = skewers_r
            # Check iterations
            if maxit > 0 and it == maxit:
                stop = True
                idx = r
            else:
                it = it + 1
    # Endmembers
    E = np.zeros((idx.shape[0], p1), dtype=np.float32)
    for j in range(idx.shape[0]):
        E[j] = M[idx[j]]

    return E, idx


def PPI(M, q, numSkewers, ini_skewers=None):
    """
    Performs the pixel purity index algorithm for endmember finding.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data ((m x n) x p).

        q: `int`
            Number of endmembers to find.

        numSkewers: `int`
            Number of "skewer" vectors to project data onto.

        ini_skewers: `numpy array [default None]`
            You can generate skewers from another source.

    Returns: `numpy array`
        Recovered endmembers (N x p).
    """
    M = M.T # temporary solution
    M = np.matrix(M, dtype=np.float32)
    # rows are bands
    # columns are signals
    p, N = M.shape

    # Remove mean from data
    u = np.transpose(np.transpose(M).mean(axis=0))
    Mm = M - np.kron(np.ones((1,N)), u)

    #Generate skewers
    if ini_skewers == None:
        skewers = np.random.rand(p, numSkewers)
    else:
        skewers = ini_skewers

    votes = np.zeros((N, 1))

    for kk in range(numSkewers):
        # skewers[:,kk] is already a row
        tmp = abs(skewers[:,kk]*Mm)
        idx = np.argmax(tmp)
        votes[idx] = votes[idx] + 1

    max_idx = np.argsort(votes, axis=None)
    # the last right idx..s at the max_idx list are
    # those with the max votes
    end_member_idx = max_idx[-q:][::-1]
    U = M[:, end_member_idx]

    return np.array(U).T, end_member_idx
