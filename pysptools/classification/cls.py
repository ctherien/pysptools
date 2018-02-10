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
# cls.py - This file is part of the PySptools package.
#


"""
SAM_classifier, SID_classifier, NormXCorr_classifier functions
"""

from __future__ import division

import numpy as np
import scipy.stats as ss


def _single_value_min(data, threshold):
    """
    Use a threshold to extract the minimum value along
    the data y axis.
    """
    min_vec = np.min(data, axis=1)
    amin = np.min(data)
    amax = np.max(data)
    limit = amin + (amax - amin) * threshold
    min_mask = min_vec < limit
    argmin = np.argmin(data, axis=1)
    return (argmin + 1) * min_mask


def _multiple_values_min(data, threshold):
    """
    Use a threshold to extract the minimum value along
    the data y axis.
    A new threshold value is used for each value of y.
    """
    data_min = np.zeros((data.shape[0], data.shape[1]), dtype=np.float)
    for i in range(data.shape[1]):
        amin = np.min(data[:,i])
        amax = np.max(data[:,i])
        limit = amin + (amax - amin) * threshold[i]
        min_mask = data[:,i] < limit
        max_mask = data[:,i] >= limit
        # The guard of value 10 is greater than
        # the largest radian value (6.3).
        rmax = max_mask * 10
        data_min[:,i] = min_mask * data[:,i] + rmax
    min_vec = np.min(data_min, axis=1)
    min_mask = min_vec < 10
    argmin = np.argmin(data_min, axis=1)
    return (argmin + 1) * min_mask


def _single_value_max(corr, threshold):
    """
    Use a threshold to extract the maximum value along
    the corr y axis.
    """
    max_vec = np.max(corr, axis=1)
    cmin = np.min(max_vec)
    cmax = np.max(max_vec)
    limit = cmax - (cmax - cmin) * threshold
    max_mask = max_vec > limit
    argmax = np.argmax(corr, axis=1)
    return (argmax + 1) * max_mask


def _multiple_values_max(corr, threshold):
    """
    Use a threshold to extract the minimum value along
    the corr y axis.
    A new threshold value is used for each value of y.
    """
    corr_max = np.zeros((corr.shape[0], corr.shape[1]), dtype=np.float)
    for i in range(corr.shape[1]):
        cmin = np.min(corr[:,i])
        cmax = np.max(corr[:,i])
        limit = cmax - (cmax - cmin) * threshold[i]
        min_mask = corr[:,i] <= limit
        max_mask = corr[:,i] > limit
        # for NormXCorr_classifier the delta is between [-1..1],
        # a guard with a -10 value is ok here.
        rmin = min_mask * -10
        corr_max[:,i] = max_mask * corr[:,i] + rmin
    max_vec = np.max(corr_max, axis=1)
    max_mask = max_vec > -10
    argmax = np.argmax(corr_max, axis=1)
    return (argmax + 1) * max_mask


# For the SAM_classifier function, if you have these errors messages:
"""
C:\somepath\pysptools\classifiers\cls.py:98: RuntimeWarning: invalid value encountered in arccos
  angles = np.arccos(np.round(sum_T_R / mul_T_R, _round_threshold))
C:\somepath\pysptools\classifiers\cls.py:19: RuntimeWarning: invalid value encountered in less
  min_mask = min_vec < limit
"""
# ajust the _round_threshold parameter accordingly,
# a smaller value avoid these errors.
_round_threshold = 7


def SAM_classifier(M, E, threshold):
    """
    Classify a HSI cube M using the spectral angle mapper
    and a spectral library E
    This function is part of the SAM class.

    Parameters
        M : numpy array
          a HSI cube ((m*n) x p)

        E : numpy array
          a spectral library (N x p)

    Returns : numpy array
          a class map ((m*n))
    """
    def norm_array(m):
        res = np.zeros(m.shape[0])
        for i in range(m.shape[0]):
            res[i] = np.dot(m[i], m[i])
        return np.sqrt(res)

    TNA = norm_array(M)
    RNA = norm_array(E)
    sum_T_R = np.dot(E, M.T).T
    mul_T_R = np.ndarray((TNA.shape[0], RNA.shape[0]), dtype=np.float)
    for i in range(TNA.shape[0]):
        mul_T_R[i] = np.multiply(TNA[i],RNA)
    # read above for _round_threshold
    angles = np.arccos(np.round(sum_T_R / mul_T_R, _round_threshold))
    if isinstance(threshold, float):
        cmap = _single_value_min(angles, threshold)
    elif isinstance(threshold, list):
        cmap = _multiple_values_min(angles, threshold)
    else:
        return np.argmin(angles, axis=1), angles
    return cmap, angles


def SID_classifier(M, E, threshold):
    """
    Classify a HSI cube M using the spectral information divergence
    and a spectral library E.
    This function is part of the NormXCorr class.

    Parameters
        M : numpy array
          a HSI cube ((m*n) x p)

        E : numpy array
          a spectral library (N x p)

    Returns : numpy array
          a class map ((m*n))
    """
    def prob_vector_array(m):
        pv_array = np.ndarray(shape=m.shape, dtype=np.float32)
        sum_m = np.sum(m, axis=1)
        pv_array[:] = (m.T / sum_m).T
        return pv_array + np.spacing(1)

    mn = M.shape[0]
    N = E.shape[0]
    p = prob_vector_array(M)
    q = prob_vector_array(E)
    sid = np.ndarray((mn, N), dtype=np.float)
    for i in range(mn):
        pq = q[0:,:] * np.log(q[0:,:] / p[i,:])
        pp = p[i,:] * np.log(p[i,:] / q[0:,:])
        sid[i,:] = np.sum(pp[0:,:] + pq[0:,:], axis=1)
    if isinstance(threshold, float):
        cmap = _single_value_min(sid, threshold)
    elif isinstance(threshold, list):
        cmap = _multiple_values_min(sid, threshold)
    else:
        return np.argmin(sid, axis=1), sid
    return cmap, sid


def NormXCorr_classifier(M, E, threshold):
    """
    Classify a HSI cube M using the normalized cross correlation
    and a spectral library E.
    This function is part of the NormXCorr class.

    Parameters
        M : numpy array
          a HSI cube ((m*n) x p)

        E : numpy array
          a spectral library (N x p)

    Returns : numpy array
          a class map ((m*n))
    """
    def tstd_array(m):
        std_array = np.ndarray(shape=m.shape[0], dtype=np.float32)
        for i in range(m.shape[0]):
            std_array[i] = ss.tstd(m[i])
        return std_array

    def signal_minus_mean_array(m):
        smm_array = np.ndarray(shape=m.shape, dtype=np.float32)
        for i in range(m.shape[0]):
            smm_array[i] = m[i] - np.mean(m[i])
        return smm_array

    mn, s = M.shape
    N = E.shape[0]
    norm = 1./(s-1)
    E_std = tstd_array(E)
    E_minus_mean = signal_minus_mean_array(E)
    M_std = tstd_array(M)
    M_minus_mean = signal_minus_mean_array(M)
    corr = np.ndarray((mn, N), dtype=np.float)
    for i in range(mn):
        mean_prod = M_minus_mean[i,:] * E_minus_mean[0:,:]
        std_prod = M_std[i] * E_std[0:]
        corr[i] = (np.sum(mean_prod[0:], axis=1) / std_prod[0:]) * norm
    if isinstance(threshold, float):
        cmap = _single_value_max(corr, threshold)
    elif isinstance(threshold, list):
        cmap = _multiple_values_max(corr, threshold)
    else:
        return np.argmax(corr, axis=1), corr
    return cmap, corr
