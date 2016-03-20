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
# dnoise.py - This file is part of the PySptools package.
#

from __future__ import division

import numpy as np
import os.path as osp
import pysptools.util as util


def whiten(M):
    """
    Whitens a HSI cube. Use the noise covariance matrix to decorrelate
    and rescale the noise in the data (noise whitening).
    Results in transformed data in which the noise has unit variance
    and no band-to-band correlations.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

    Returns: `numpy array`
        Whitened HSI data (N x p).

    Reference:
        Krizhevsky, Alex, Learning Multiple Layers of Features from
        Tiny Images, MSc thesis, University of Toronto, 2009.
        See Appendix A.
    """
    sigma = util.cov(M)
    U,S,V = np.linalg.svd(sigma)
    S_1_2 = S**(-0.5)
    S = np.diag(S_1_2.T)
    Aw = np.dot(V, np.dot(S, V.T))
    return np.dot(M, Aw)
