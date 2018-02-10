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
# test_detect.py - This file is part of the PySptools package.
#

"""
The following functions are tested:
    MatchedFilter
    ACE
"""

from __future__ import print_function

import os
import os.path as osp
import numpy as np
import cProfile, pstats

try:
    import spectral.io.envi as envi
except ImportError:
    pass

import pysptools.util as util
import pysptools.detection as detect


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


def load_signal_to_detect(detect_path, detect_hdr_name):
    dfin = osp.join(detect_path,detect_hdr_name)
    sli = envi.open(dfin)
    y = sli.spectra[0,:].tolist()
    return np.array(y)


def load_signal_to_detect_P3(detect_path, detect_hdr_name):
    import json
    dfin = osp.join(detect_path,detect_hdr_name)
    with open(dfin, 'r') as content_file:
        y = np.array(json.loads(content_file.read()))
    return y[0,:]


def load_spec_lib(lib_path, lib_hdr_name):
    dfin = osp.join(lib_path, lib_hdr_name)
    sli = envi.open(dfin)
    return sli.spectra


def test_MatchedFilter(data, y, result_path):
    print('Testing MatchedFilter')
    mf = detect.MatchedFilter()
    pr = profile()
    target_map = mf.detect(data, y)
    #target_map = mf.detect(data, y, threshold='string')
    #target_map = mf.detect(data, np.zeros((2,2)), threshold=0.2)
    stat(pr)
    print(str(mf))
    mf.plot(result_path, suffix='test')
    #mf.plot(result_path, suffix=0.1)
    #mf.plot(result_path, whiteOnBlack='allo', suffix='test')


def test_ACE(data, y, result_path):
    print('Testing ACE')
    ace = detect.ACE()
    pr = profile()
    target_map = ace.detect(data, y)
    stat(pr)
    print(str(ace))
    ace.plot(result_path, whiteOnBlack=False, suffix='test')


def test_CEM(data, y, result_path):
    print('Testing CEM')
    cem = detect.CEM()
    pr = profile()
    target_map = cem.detect(data, y)
    stat(pr)
    print(str(cem))
    cem.plot(result_path, suffix='test')


def test_GLRT(data, y, result_path):
    print('Testing GLRT')
    glrt = detect.GLRT()
    pr = profile()
    target_map = glrt.detect(data, y)
    stat(pr)
    print(str(glrt))
    glrt.plot(result_path, whiteOnBlack=False, suffix='test')


def test_OSP(data, U, y, result_path):
    print('Testing OSP')
    osp = detect.OSP()
    pr = profile()
    target_map = osp.detect(data, U, y)
    #target_map = osp.detect(data, 'string', y)
    stat(pr)
    print(str(osp))
    osp.plot(result_path, suffix='test')


def tests():
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    # load the cube
    sample = 'samson_part.hdr'
    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)

    # load the spectrum to detect
    to_detect_hdr_name = 'white_roof.hdr'
    y = load_signal_to_detect(data_path, to_detect_hdr_name)

    # load some background pixels needed by OSP
    background = 'bground1.hdr'
    lib_file = osp.join(data_path, background)
    U, info = util.load_ENVI_spec_lib(lib_file)

    test_MatchedFilter(data, y, result_path)
    test_ACE(data, y, result_path)
    test_CEM(data, y, result_path)
    test_GLRT(data, y, result_path)
    test_OSP(data, U, y, result_path)


if __name__ == '__main__':
    import sys
    print(sys.version_info)
    tests()
