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
# envi.py - This file is part of the PySptools package.
#

"""
load_ENVI_file, load_ENVI_spec_lib functions
"""


import numpy as np
import spectral.io.envi as envi

def load_ENVI_file(file_name):
    """
    Load the data and the header from an ENVI file.
    It use the SPy (spectral) library. At 'file_name' give the envi header file name.

    Parameters:
        file_name: `path string`
            The complete path to the file to load. Use the header file name.

    Returns: `tuple`
        data: `numpy array`
            A (m x n x p) HSI cube.

        head: `dictionary`
            Starting at version 0.13.1, the ENVI file header
     """
    img = envi.open(file_name)
    head = envi.read_envi_header(file_name)
    return np.array(img.load()), head


def load_ENVI_spec_lib(file_name):
    """
    Load an ENVI.sli file.

    Parameters:
        file_name: `path string`
            The complete path to the library file to load.

    Returns: `numpy array`
        A (n x p) HSI cube.

        head: `dictionary`
            Starting at version 0.13.1, the ENVI file header

    """
    sli = envi.open(file_name)
    head = envi.read_envi_header(file_name)
    return sli.spectra, head
