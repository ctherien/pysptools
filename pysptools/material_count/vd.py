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
# vd.py - This file is part of the PySptools package.
#

"""
HfcVd function
"""

from __future__ import division

import numpy as np
import scipy as sp
import scipy.stats as ss

def est_noise(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w=np.zeros((L,N), dtype=np.float)
        RR=np.dot(r,r.T)
        RRi = np.linalg.pinv(RR+small*np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
            RRa = RR[:,i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0,i]=0;
            w[i,:] = r[i,:] - np.dot(beta,r)
        Rw = np.diag(np.diag(np.dot(w,w.T) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    #verb = 'poisson'
    if noise_type == 'poisson':
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = np.sqrt(x)*u*2
        Rw = np.dot(w,w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T


def hysime(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: `numpy array`
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: `numpy array`
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: `numpy array`
            noise correlation matrix (p x p)

    Returns: `tuple integer, numpy array`
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Copyright:
        Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    y=y.T
    n=n.T
    Rn=Rn.T
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n;

    Ry = np.dot(y, y.T) / N
    Rx = np.dot(x, x.T) / N
    E, dx, V = np.linalg.svd(Rx)

    Rn = Rn+np.sum(np.diag(Rx))/L/10**5 * np.eye(L)
    Py = np.diag(np.dot(E.T, np.dot(Ry,E)))
    Pn = np.diag(np.dot(E.T, np.dot(Rn,E)))
    cost_F = -Py + 2 * Pn
    kf = np.sum(cost_F < 0)
    ind_asc = np.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, Ek # Ek.T ?


# Comments on using complex number:
#
# Use only scipy and numpy functions for a correct use of complex number.
#
# scipy.sqrt() deal by the book with complex number,
# it's more tricky when using math and numpy modules.
#


def HfcVd(M, far='default'):
    """
    Computes the vitual dimensionality (VD) measure for an HSI
    image for specified false alarm rates.  When no false alarm rate(s) is
    specificied, the following vector is used: 1e-3, 1e-4, 1e-5.
    This metric is used to estimate the number of materials in an HSI scene.

    Parameters:
       M: `numpy array`
           HSI data as a 2D matrix (N x p).

       far: `list [default default]`
           False alarm rate(s).

    Returns: python list
           VD measure, number of materials estimate.

    References:
        C.-I. Chang and Q. Du, "Estimation of number of spectrally distinct
        signal sources in hyperspectral imagery," IEEE Transactions on
        Geoscience and Remote Sensing, vol. 43, no. 3, mar 2004.

        J. Wang and C.-I. Chang, "Applications of independent component
        analysis in endmember extraction and abundance quantification for
        hyperspectral imagery," IEEE Transactions on Geoscience and Remote
        Sensing, vol. 44, no. 9, pp. 2601-1616, sep 2006.
    """
    N, numBands = M.shape

    # calculate eigenvalues of covariance and correlation between bands
    lambda_cov = np.linalg.eig(np.cov(M.T))[0] # octave: cov(M')
    lambda_corr = np.linalg.eig(np.corrcoef(M.T))[0] # octave: corrcoef(M')
    # not realy needed:
    lambda_cov = np.sort(lambda_cov)[::-1]
    lambda_corr = np.sort(lambda_corr)[::-1]

    if far == 'default':
        far = [10**-3, 10**-4, 10**-5]
    else:
        far = [far]

    numEndmembers_list = []
    for y in range(len(far)):
        numEndmembers = 0
        pf = far[y]
        for x in range(numBands):
            sigmaSquared = (2.*lambda_cov[x]/N) + (2.*lambda_corr[x]/N) + (2./N)*lambda_cov[x]*lambda_corr[x]
            sigma = sp.sqrt(sigmaSquared)
            tau = -ss.norm.ppf(pf, 0, abs(sigma))
            if (lambda_corr[x]-lambda_cov[x]) > tau:
                numEndmembers += 1
        numEndmembers_list.append(numEndmembers)
    return numEndmembers_list

