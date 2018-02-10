#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2015, Christian Therien
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
# misc.py - This file is part of the PySptools package.
#


import numpy as np
from scipy import io


def load_mat_file(fname):
    """
    Load a MATLAB file (a 3D array only)
    """
    m_dic = io.loadmat(fname)
    return m_dic['ref']


def shrink(M):
    """
    Reduce a hyperspectral image by half.
    """
    h, w, numBands = M.shape
    del_rows = [x for x in range(h) if x%2]
    del_columns = [x for x in range(w) if x%2]
    rows_less = np.delete(M, del_rows, 0)
    columns_less = np.delete(rows_less, del_columns, 1)
    return columns_less
