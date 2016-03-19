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
# detect.py - This file is part of the PySptools package.
#


"""
MatchedFilter, ACE, CEM, GLRT, OSP functions
"""

from __future__ import division

import numpy as np
import scipy.linalg as lin


def ACE(M, t):
    """
    Performs the adaptive cosin/coherent estimator algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
      X Jin, S Paswater, H Cline.  "A Comparative Study of Target Detection
      Algorithms for Hyperspectral Imagery."  SPIE Algorithms and Technologies
      for Multispectral, Hyperspectral, and Ultraspectral Imagery XV.  Vol
      7334.  2009.
    """
    N, p = M.shape
    # Remove mean from data
    u = M.mean(axis=0)
    M = M - np.kron(np.ones((N, 1)), u)
    t = t - u;

    R_hat = np.cov(M.T)
    G = lin.inv(R_hat)

    results = np.zeros(N, dtype=np.float32)
    ##% From Broadwater's paper
    ##%tmp = G*S*inv(S.'*G*S)*S.'*G;
    tmp = np.array(np.dot(t.T, np.dot(G, t)))
    dot_G_M = np.dot(G, M[0:,:].T)
    num = np.square(np.dot(t, dot_G_M))
    for k in range(N):
        denom = np.dot(tmp, np.dot(M[k], dot_G_M[:,k]))
        results[k] = num[k] / denom
    return results


def MatchedFilter(M, t):
    """
    Performs the matched filter algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
      X Jin, S Paswater, H Cline.  "A Comparative Study of Target Detection
     Algorithms for Hyperspectral Imagery."  SPIE Algorithms and Technologies
     for Multispectral, Hyperspectral, and Ultraspectral Imagery XV.  Vol
     7334.  2009.
    """

    N, p = M.shape
    # Remove mean from data
    u = M.mean(axis=0)
    M = M - np.kron(np.ones((N, 1)), u)
    t = t - u;

    R_hat = np.cov(M.T)
    G = lin.inv(R_hat)

    tmp = np.array(np.dot(t.T, np.dot(G, t)))
    w = np.dot(G, t)
    return np.dot(w, M.T) / tmp


def CEM(M, t):
    """
    Performs the constrained energy minimization algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
        Qian Du, Hsuan Ren, and Chein-I Cheng. A Comparative Study of
        Orthogonal Subspace Projection and Constrained Energy Minimization.
        IEEE TGRS. Volume 41. Number 6. June 2003.
    """
    def corr(M):
        p, N = M.shape
        return np.dot(M, M.T) / N

    N, p = M.shape
    R_hat = corr(M.T)
    Rinv = lin.inv(R_hat)
    denom = np.dot(t.T, np.dot(Rinv, t))
    t_Rinv = np.dot(t.T, Rinv)
    return np.dot(t_Rinv , M[0:,:].T) / denom


def GLRT(M, t):
    """
    Performs the generalized likelihood test ratio algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
        T F AyouB, "Modified GLRT Signal Detection Algorithm," IEEE
        Transactions on Aerospace and Electronic Systems, Vol 36, No 3, July
        2000.
    """
    N, p = M.shape

    # Remove mean from data
    u = M.mean(axis=0)
    M = M - np.kron(np.ones((N, 1)), u)
    t = t - u;

    R = lin.inv(np.cov(M.T))
    results = np.zeros(N, dtype=np.float)

    t_R_t_dot = np.dot(t, np.dot(R, t.T))
    for k in range(N):
        x = M[k]
        R_x_dot = np.dot(R, x)
        num = np.dot(t, R_x_dot)**2
        denom = t_R_t_dot * (1 + np.dot(x.T, R_x_dot))
        results[k] = num / denom
    return results


def OSP(M, E, t):
    """
    Performs the othogonal subspace projection algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        E: `numpy array`
            2d matrix of background endmebers (p x q).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
        Qian Du, Hsuan Ren, and Chein-I Cheng. "A Comparative Study of
        Orthogonal Subspace Projection and Constrained Energy Minimization."
        IEEE TGRS. Volume 41. Number 6. June 2003.
    """
    N, p = M.shape
    P_U = np.eye(p, dtype=np.float) - np.dot(E.T, lin.pinv(E.T))
    tmp = np.dot(t.T, np.dot(P_U, t))
    nu = np.zeros(N, dtype=np.float)
    for k in range(N):
        nu[k] = np.dot(t.T, np.dot(P_U, M[k])) / tmp
    return nu
