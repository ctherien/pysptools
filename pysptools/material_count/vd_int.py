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
# vd_int.py - This file is part of the PySptools package.
#

"""
HySime class
HfcVd class
"""

import numpy as np
from . import vd
from .inval import *


class HySime(object):
    """ Hyperspectral signal subspace identification by minimum error. """

    def __init__(self):
        self.kf = None
        self.Ek = None

    @CountInputValidation1('HySime')
    def count(self, M):
        """
        Hyperspectral signal subspace estimation.

        Parameters:
            M: `numpy array`
                Hyperspectral data set (each row is a pixel)
                with ((m*n) x p), where p is the number of bands
                and (m*n) the number of pixels.

        Returns: `tuple integer, numpy array`
            * kf signal subspace dimension
            * Ek matrix which columns are the eigenvectors that span
              the signal subspace.

        Reference:
            Bioucas-Dias, Jose M., Nascimento, Jose M. P., 'Hyperspectral Subspace Identification',
            IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 46, NO. 8, AUGUST 2008.

        Copyright:
            Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
            For any comments contact the authors.
        """
        h, w, numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        w, Rw = vd.est_noise(Mr)
        self.kf, self.Ek = vd.hysime(Mr, w, Rw)
        return self.kf, self.Ek


class HfcVd(object):
    """
    Computes the vitual dimensionality (VD) measure for an HSI
    image for specified false alarm rates.
    """

    def __init__(self):
        self.vd = None

    @CountInputValidation2('HfcVd')
    def count(self, M, far='default', noise_whitening=False):
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

           noise_whitening: `boolean [default False]`
               If True noise whitening is applied before calling HfcVd,
               doing a NWHFC.

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
        import pysptools.noise as ns
        h, w, numBands = M.shape
        Mr = np.reshape(M, (w*h, numBands))
        if noise_whitening == True:
            Mr = ns.whiten(Mr)
        self.vd = vd.HfcVd(Mr, far)
        return self.vd
