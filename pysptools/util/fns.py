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
# fns.py - This file is part of the PySptools package.
#

"""
corr, cov functions
"""

from __future__ import division

import numpy as np

def corr(M):
    """
    Compute the sample autocorrelation matrix of a 2D matrix.

    Parameters:
      M: `numpy array`
        2d matrix of HSI data (N x p)

    Returns: `numpy array`
        Sample autocorrelation matrix.
    """
    N = M.shape[0]
    return np.dot(M, M.T) / N


def cov(M):
    """
    Compute the sample covariance matrix of a 2D matrix.

    Parameters:
      M: `numpy array`
        2d matrix of HSI data (N x p)

    Returns: `numpy array`
        sample covariance matrix
    """
    N = M.shape[0]
    u = M.mean(axis=0)
    M = M - np.kron(np.ones((N, 1)), u)
    C = np.dot(M.T, M) / (N-1)
    return C
