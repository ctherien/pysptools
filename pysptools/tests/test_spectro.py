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
# test_spectro.py - This file is part of the PySptools package.
#

"""
The following functions are tested:
    distance.SAM
    distance.SID
    distance.chebyshev
    distance.NormXCorr

This class is tested:
    USGS06SpecLib
        get_substance method
        distance_match method
"""

from __future__ import print_function

import os
import pysptools.spectro as spectro


def get_biotite_WS660(lib_name, file_type):
    """
    """
    if file_type == 'ENVI': rd = spectro.EnviReader(lib_name)
    if file_type == 'JSON': rd = spectro.JSONReader(lib_name)
    lib = spectro.USGS06SpecLib(rd)
    for spectrum, sample_id, descrip, idx in lib.get_substance('Biotite', 'WS660'):
        return spectrum, idx


def search_biotite(lib_name, file_type, biotite, dist):
    """
    """
    if file_type == 'ENVI': rd = spectro.EnviReader(lib_name)
    if file_type == 'JSON': rd = spectro.JSONReader(lib_name)
    lib = spectro.USGS06SpecLib(rd)
    match_spectrum, where = lib.distance_match(biotite, distfn=dist)
    return where


def tests():
    data_path = os.environ['PYSPTOOLS_USGS']

    hdr_name = 's06av95a_envi.hdr'
    lib_name = os.path.join(data_path, hdr_name)

    biotite, idx = get_biotite_WS660(lib_name, 'ENVI')
    print('Official WS660 biotite at index:', idx)

    found = search_biotite(lib_name, 'ENVI', biotite, 'SAM')
    assert found == idx, "Error in search_biotite with SAM"
    print('Found WS660 biotite with SAM at index:', found)

    found = search_biotite(lib_name, 'ENVI', biotite, 'SID')
    assert found == idx, "Error in search_biotite with SID"
    print('Found WS660 biotite with SID at index:', found)

    found = search_biotite(lib_name, 'ENVI', biotite, 'chebyshev')
    assert found == idx, "Error in search_biotite with chebyshev"
    print('Found WS660 biotite with chebyshev at index:', found)

    found = search_biotite(lib_name, 'ENVI', biotite, 'NormXCorr')
    assert found == idx, "Error in search_biotite with NormXCorr"
    print('Found WS660 biotite with NormXCorr at index:', found)


if __name__ == '__main__':
    tests()
