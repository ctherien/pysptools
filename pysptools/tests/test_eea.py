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
# test_eea.py - This file is part of the PySptools package.
#

"""
The following functions are tested:
    NFINDR
    ATGP
    PPI
    UCLS
    FCLS
    NNLS
"""

from __future__ import print_function


import os
import os.path as osp
import cProfile, pstats

import pysptools.util as util
import pysptools.eea as eea
import pysptools.abundance_maps as amp


_doProfile = False
def profile():
    if _doProfile == True:
        pr = cProfile.Profile()
        pr.enable()
        return pr


def stat(pr):
    if _doProfile == True:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats('time')
        ps.print_stats()


def ROI(data, path):
    r = util.ROIs(data.shape[0], data.shape[1])
    # Some roi
    r.add('Area', {'rec': (30,30,100,100)})
    r.plot(path)
    return r


def parse_ENVI_header(fname, head):
    # Parse a ENVI header and fill up
    # the axes dictionary. This later
    # is needed by the PPI ... plot method.
    axes = {}
    if fname == '92AV3C.hdr' or fname == '92AV3C':
        axes['wavelength'] = head['wavelength']
        axes['x'] = 'Wavelength (Unknown)'
        axes['y'] = 'Reflectance'
    return axes


def test_PPI(data, wvl, mask, path):
    print('Testing PPI')
    ppi = eea.PPI()
    # The format of the data is always (m x n x p),
    # for each class interface
    # the format of U is always (N x p)
    pr = profile()
    #U = ppi.extract(data, 4, normalize=True)
    U = ppi.extract(data, 4, normalize=True, mask=mask)
    print(str(ppi))
    #U = ppi.extract(data, 'string', normalize=True)
    #U = ppi.extract(data, 4, normalize=None)
    stat(pr)
    print('  End members indexes:', ppi.get_idx())
    ppi.plot(path, axes=wvl, suffix='test1')
    #ppi.plot(path, axes=1.0, suffix='test1.1')
    ppi.plot(path, suffix='test2')
    U = U[[0,1],:]
    test_amap(data, U, 'PPI', path, mask, amaps='UCLS')
    test_amap(data, U, 'PPI', path, mask, amaps='NNLS')
    test_amap(data, U, 'PPI', path, mask, amaps='FCLS')


def test_NFINDR(data, wvl, mask, path):
    print('Testing NFINDR')
    nfindr = eea.NFINDR()
    pr = profile()
#    U = nfindr.extract(data, 4, normalize=1, ATGP_init=True)
    #U = nfindr.extract(data, 4, maxit=5, normalize=False, ATGP_init=True)
#    U = nfindr.extract(data, 8, maxit=5, normalize=False, ATGP_init=True, mask=mask)
    U = nfindr.extract(data, 8, maxit=5, normalize=False, ATGP_init=False, mask=mask)
#    U = nfindr.extract(data, 4, maxit=5, normalize=True, ATGP_init=True)
    stat(pr)
    print(str(nfindr))
    print('  Iterations:', nfindr.get_iterations())
    print('  End members indexes:', nfindr.get_idx())
    nfindr.plot(path, axes=wvl, suffix='test1')
    nfindr.plot(path, suffix='test2')
    U = U[[0,1],:]
    test_amap(data, U, 'NFINDR', path, mask, amaps='UCLS')
    test_amap(data, U, 'NFINDR', path, mask, amaps='NNLS')
    test_amap(data, U, 'NFINDR', path, mask, amaps='FCLS')


def test_ATGP(data, wvl, mask, path):
    print('Testing ATGP')
    atgp = eea.ATGP()
    pr = profile()
    #U = atgp.extract(data, 8, normalize=True)
    U = atgp.extract(data, 8, normalize=True, mask=mask)
    stat(pr)
    print(str(atgp))
    print('  End members indexes:', atgp.get_idx())
    atgp.plot(path, axes=wvl, suffix='test1')
    atgp.plot(path, suffix='test2')
    U = U[[0,1],:]
    test_amap(data, U, 'ATGP', path, mask, amaps='FCLS')


def test_FIPPI(data, wvl, mask, path):
    print('Testing FIPPI')
    fippi = eea.FIPPI()
    pr = profile()
    U = fippi.extract(data, 4, 1, normalize=True, mask=mask)
    #U = fippi.extract(data, 4, 1, normalize=True)
    print(str(fippi))
    stat(pr)
    print('  End members indexes:', fippi.get_idx())
    fippi.plot(path, axes=wvl, suffix='test1')
    fippi.plot(path, suffix='test2')
    #U = U[[0,1],:]
    test_amap(data, U, 'FIPPI', path, mask, amaps='NNLS')


def test_amap(data, U, umix_source, path, mask, amaps=None):
    # if you normalize at the extract step, you should
    # normalize at abundance map generation
    # ... and the opposite
    if amaps == None:
        test_UCLS(data, U, umix_source, mask, path)
        test_NNLS(data, U, umix_source, mask, path)
        test_FCLS(data, U, umix_source, mask, path)
    else:
        if 'UCLS' in amaps: test_UCLS(data, U, umix_source, mask, path)
        if 'NNLS' in amaps: test_NNLS(data, U, umix_source, mask, path)
        if 'FCLS' in amaps: test_FCLS(data, U, umix_source, mask, path)


def test_UCLS(data, U, umix_source, mask, path):
    import numpy as np
    print('  Testing UCLS')
    ucls = amp.UCLS()
    pr = profile()
    amap = ucls.map(data, U, normalize=True, mask=mask)
    #amap = ucls.map(data, np.zeros((2,2,2,2)), normalize=True)
    #amap = ucls.map(data, U, normalize=3)
    stat(pr)
    print(str(ucls))
    #ucls.plot(path, suffix=12)
    #ucls.plot(path, mask=[1,2], suffix=umix_source)
    ucls.plot(path, suffix=umix_source)
    ucls.plot(path, mask=mask, colorMap='jet', suffix=umix_source+'_mask')
    ucls.plot(path, interpolation='spline36', suffix=umix_source+'_spline36')
    # throw a warning
    ucls.plot(path, mask=mask, interpolation='spline36', columns= 2, suffix=umix_source+'_spline36')
    ucls.plot(path, mask=mask, interpolation='spline36', suffix=umix_source+'_spline36')


def test_NNLS(data, U, umix_source, mask, path):
    print('  Testing NNLS')
    nnls = amp.NNLS()
    pr = profile()
    amap = nnls.map(data, U, normalize=True)
    stat(pr)
    print(str(nnls))
    nnls.plot(path, colorMap='jet', suffix=umix_source)
    nnls.plot(path, mask=mask, colorMap='jet', suffix=umix_source+'_mask')
    nnls.plot(path, interpolation='spline36', suffix=umix_source+'_spline36')


def test_FCLS(data, U, umix_source, mask, path):
    print('  Testing FCLS')
    fcls = amp.FCLS()
    pr = profile()
    amap = fcls.map(data, U, normalize=True, mask=mask)
    stat(pr)
#    fcls.plot(path, columns=2, suffix=umix_source)
    print(str(fcls))
    fcls.plot(path, suffix=umix_source)
#    fcls.plot(path, mask=mask, colorMap='jet', suffix=umix_source+'_mask')
#    fcls.plot(path, interpolation='spline36', suffix=umix_source+'_spline36')


def tests():
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    sample = '92AV3C.hdr'
    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)
    axes = parse_ENVI_header(sample, header)

    roi = ROI(data, result_path)

    m = roi.get_mask()
    test_PPI(data, axes, m, result_path)
    test_ATGP(data, axes, m, result_path)
    test_FIPPI(data, axes, m, result_path)
    test_NFINDR(data, axes, m, result_path)


if __name__ == '__main__':
    import sys
    print(sys.version_info)
    tests()
