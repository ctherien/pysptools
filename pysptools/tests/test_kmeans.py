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
# test_kmeans.py - This file is part of the PySptools package.
#

"""
The following class is tested:
    KMeans
"""

import os
import os.path as osp
import pysptools.sklearn as skl
import pysptools.util as util


def tests():
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    sample = '92AV3C.hdr'

    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)
    #data = np.fliplr(data)

    km = skl.KMeans()
    km.predict(data, 5)
#    km.plot(result_path, colorMap='jet')
    km.plot(result_path, interpolation=None, colorMap='jet')


if __name__ == '__main__':
    tests()
