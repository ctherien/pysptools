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
# test_HyperSVC.py - This file is part of the PySptools package.
#

from __future__ import print_function

import os
import os.path as osp

import pysptools.skl as skl
import pysptools.util as util

def remove_bands(M):
    """
    Remove the bands with atmospheric
    scattering.
    Remove:
        [0..4]
        [102..110]
        [148..169]
        [211..end]
    """
    p1 = list(range(5,102))
    p2 = list(range(111,148))
    p3 = list(range(170,211))
    Mp = M[:,:,p1+p2+p3]
    return Mp

# Do a supervised clustering
def test_HyperSVC(data, result_path):
    r = util.ROIs(data.shape[0], data.shape[1])
    r.add('Alfalfa', {'poly': ((67,98),(73,98),(75,101),(70,101))})
    r.add('Corn-notill', {'rec': (33,31,41,56)})
    r.add('Corn-min', {'rec': (63,6,71,21)}, {'rec': (128,20,134,46)})
    r.add('Corn', {'poly': ((35,7),(35,5),(48,10),(48,23),(45,22),(44,16),(35,10),(35,5))})
    r.add('Grass/Pasture', {'rec': (75,4,85,21)})
    r.add('Grass/Trees', {'rec': (48,28,70,35)})
    r.add('Grass/pasture-mowed', {'rec': (73,109,78,112)})
    r.add('Hay-windrowed', {'rec': (39,124,59,138)})
    r.add('Soybeans-notill', {'rec': (42,78,63,92)})
    r.add('Soybean-min-till', {'rec': (78,34,111,45)}, {'rec': (3,112,17,117)}, {'rec': (80,51,95,71)})
    r.add('Soybean-clean', {'rec': (52,5,58,24)})
    r.add('Wheat', {'rec': (119,26,124,46)})
    r.add('Woods', {'rec': (121,91,137,121)})
    r.add('Stone-steel towers', {'poly': ((14,47),(23,44),(24,49),(16,52),(14,47))})

    # These one don't perform well
    #r.add('Oats', {'rec': (63,23,71,24)})
    #r.add('Bldg-Grass-Tree-Drives', {'rec': (18,27,27,34)})

    r.plot(result_path, colorMap='Paired')

    svm = skl.HyperSVC(class_weight={0:1,1:10,2:10,3:10,4:10,5:10,6:10,7:10,8:10,9:10,10:10,11:10,12:10,13:10,14:10})
    svm.fit_rois(data, r)
    svm.classify(data)
    svm.plot(result_path, labels=r.get_labels(), interpolation=None, colorMap='Paired')


def tests():
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)
        
    sample = '92AV3C.hdr'

    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)
    data = remove_bands(data)

    test_HyperSVC(data, result_path)



if __name__ == '__main__':
    tests()
