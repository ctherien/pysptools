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
# test_vd.py - This file is part of the PySptools package.
#

"""
The following functions are tested:
    HfcVd
    HySime
"""

from __future__ import print_function

import os
import os.path as osp
import numpy as np
import scipy

import pysptools.util as util
import pysptools.material_count as cnt
import pysptools.spectro as spectro


def get_random_n_endmembers(fname, file_type, n):
    import random
    if file_type == 'ENVI': rd = spectro.EnviReader(fname)
    if file_type == 'JSON': rd = spectro.JSONReader(fname)
    lib = spectro.USGS06SpecLib(rd)
    dim = lib.get_dim()
    idx = random.sample(list(range(dim)), n)
    # 224 is the number of bands
    U = np.zeros((224, n), dtype=np.float)
    for i, j in enumerate(idx):
        U[:,i] = lib.get(j)
    # the USGS library sometimes have very small numbers that create numeric
    # instability, normalize get rid of them
    return util.normalize(U)


def dirichlet_rnd(A, dim):
    """
    Returns a matrix of random numbers chosen
    from the dirichlet distribution with parameters vector A.

    Parameters:
        A: `numpy array`
            A vector of shape parameters.

    Returns: `numpy array`
        A matrix of random numbers.
    """
    N = A.shape[0]

    x = np.zeros((dim, N), dtype=np.float)
    for i in range(N):
        x[:,i] = scipy.stats.gamma.rvs(A[i], scale=1, size=dim)

    denom = np.sum(x, axis=1)
    for i in range(N):
        x[:,i] = x[:,i] / denom
    return x


def generate_hyperspectral_data(U, p, N):
    """
    Generate a simulated hyperspectral data set.

    Parameters:
        U: `numpy array`
            USGS library subset.

        p: `int`
            Number of endmembers.

        N: `int`
            Number of pixels.

    Returns: (`numpy array`, `numpy array`)
        * x is the signal (endmembers linear mixture)
        * s abundance fractions (Nxp)
    """
    numBands, p1 = U.shape
    s = dirichlet_rnd(np.ones(p)/p, N)
    # linear mixture:
    x = np.dot(U, s.T)
    return x.T, s


def test_synthetic_hypercube(lib_file_type):
    """
    Test a synthetic hypercube made with p endmembers taken to the USGS library.
    Maybe the USGS library is not a good source of endmembers. Same mineral
    species can have a very similar signature. Picking a random subset can
    return some nearly identical signatures.

    There is no noise added.

    In general, the results are good for both HySime and HfcVd for small values of p.
    """
    print('Testing synthetic hypercube')
    data_path = os.environ['PYSPTOOLS_USGS']
    #project_path = '../'

    # USGS library
    if lib_file_type == 'ENVI': # Python 2.7
        hdr_name = 's06av95a_envi.hdr'
    if lib_file_type == 'JSON': # Python 3.3
        hdr_name = 's06av95a_envi'

    lib_name = os.path.join(data_path, hdr_name)

    # number of endmembers
    p = 4
    # get a library of endmembers
    U = get_random_n_endmembers(lib_name, lib_file_type, p)
    # cube dimension
    x_coord = 100
    y_coord = 100
    # number of pixels
    N = x_coord*y_coord

    y, s = generate_hyperspectral_data(U, p, N)
    # y is a vector of pixels, yr is the equivalent cube
    # 224 is the number of bands
    yr = np.reshape(y, (x_coord, y_coord, 224))
    # calculate kf
    hy = cnt.HySime()
    kf, Ek = hy.count(yr)
    print('  HySime kf:',kf)
    # calculate vd
    hfcvd = cnt.HfcVd()
    vd = hfcvd.count(yr)
    print('  HfcVd vd:',vd)


def test_hysime(data):
    hy = cnt.HySime()
    kf, Ek = hy.count(data)
    print('Testing HySime')
    print('  Virtual dimensionality is: k =', kf)


def test_HfcVd(data):
    hfcvd = cnt.HfcVd()
    print('Testing HfcVd')
    print('  Virtual dimensionality:', hfcvd.count(data))
    print('Testing NWHFC')
    print('  Virtual dimensionality:', hfcvd.count(data, noise_whitening=True))


def tests():
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    sample = '92AV3C.hdr'

    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)

    test_hysime(data)
    test_HfcVd(data)
    test_synthetic_hypercube('ENVI')


if __name__ == '__main__':
    tests()
