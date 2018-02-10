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
# test_pwc.py - This file is part of the PySptools package.
#

"""
The following function is tested:
    bilateral
"""

from __future__ import print_function

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pysptools.sigproc as sig

def tests():
    plt.ioff()
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    fin = open(os.path.join(data_path, 'dnagwas.txt'))
    signal_txt = fin.readlines()
    signal = [float(x) for x in signal_txt]
    z = sig.bilateral(np.array(signal), 0, 10, 25, display=1, maxiter=5)
    plt.plot(signal)
    plt.plot(z, color='r')
    if os.path.exists(result_path) == False:
        os.makedirs(result_path)
    plt.savefig(os.path.join(result_path, 'dnagwas.png'))


if __name__ == '__main__':
    import sys
    print(sys.version_info)
    tests()
