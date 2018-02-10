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
# test_sklearn.py - This file is part of the PySptools package.
#

# Tested on Python 3.5 only

from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pysptools.util as util

from pysptools.skl import (HyperEstimatorCrossVal, HyperSVC, HyperLogisticRegression,
                               HyperRandomForestClassifier, HyperKNeighborsClassifier,
                               HyperGradientBoostingClassifier, HyperAdaBoostClassifier,
                               HyperBaggingClassifier, HyperExtraTreesClassifier,
                               shape_to_XY)
from test_skl_util import HyperEstimatorTraining, DataMine, Mask, remove_bands

import warnings
warnings.filterwarnings('ignore')

def data_mine(path, hcube):
    rb_dm = DataMine(hcube, 12, 'Pine Creek')
    rb_E = rb_dm.get_endmembers()
    rb_abundances = rb_dm.get_abundances()
    #rb_dm.plot_abundances(path)
    return rb_abundances, rb_E


def mask(path, M, amaps):
    rbm = Mask('RB')
    rbm.put2(M, amaps[:,:,9], 0.05, amaps[:,:,8], 0.05)
    mask = rbm.get_mask()
    rbm.plot(path, suffix='roads_and_buildings')
    return mask
    
    
def test_HyperEstimatorCrossVal(M, mask):
    p_grid = {'C': [40,50,60], 'max_iter': [10,20]}
    ecv = HyperEstimatorCrossVal(HyperLogisticRegression, p_grid)
    X,y = shape_to_XY([M], [mask])
    ecv.fit(X,y)
    ecv.print()
    return ecv


def test_HyperLogisticRegression(path, M, mask, ecv):
    et = HyperEstimatorTraining(HyperLogisticRegression, M, mask, 0.33, 'RB', **ecv.get_best_params())
    m = et.get_model()
    m.plot(path)    


def test_HyperSVC(path, M, mask):
    param = {'C': 20, 'gamma': 0.01}
    et = HyperEstimatorTraining(HyperSVC, M, mask, 0.33, 'RB', **param)
    m = et.get_model()
    m.plot(path)    
    

def test_HyperRandomForestClassifier(path, M, mask):
    param = {'n_estimators': 5}
    et = HyperEstimatorTraining(HyperRandomForestClassifier, M, mask, 0.33, 'RB', **param)
    m = et.get_model()
    m.plot(path)    
    m.plot_feature_importances(path, n_labels=30, sort=True, suffix='test')


def test_HyperGradientBoostingClassifier(path, M, mask):
    param = {'n_estimators': 50}
    et = HyperEstimatorTraining(HyperGradientBoostingClassifier, M, mask, 0.33, 'RB', **param)
    m = et.get_model()
    m.plot(path)
    m.plot_feature_importances(path, n_labels=30, sort=True, suffix='test')


def test_HyperAdaBoostClassifier(path, M, mask):
    param = {'n_estimators': 50}
    et = HyperEstimatorTraining(HyperAdaBoostClassifier, M, mask, 0.33, 'RB', **param)
    m = et.get_model()
    m.plot(path)
    m.plot_feature_importances(path, n_labels=30, sort=False, suffix='test')


def test_HyperExtraTreesClassifier(path, M, mask):
    param = {'n_estimators': 50}
    et = HyperEstimatorTraining(HyperExtraTreesClassifier, M, mask, 0.33, 'RB', **param)
    m = et.get_model()
    m.plot(path)
    m.plot_feature_importances(path, n_labels=30, sort=True, suffix='test')


def test_HyperBaggingClassifier(path, M, mask):
    param = {'n_estimators': 50}
    et = HyperEstimatorTraining(HyperBaggingClassifier, M, mask, 0.33, 'RB', **param)
    m = et.get_model()
    m.plot(path)


def test_HyperKNeighborsClassifier(path, M, mask):
    param = {'n_neighbors': 5}
    et = HyperEstimatorTraining(HyperKNeighborsClassifier, M, mask, 0.33, 'RB', **param)
    m = et.get_model()
    m.plot(path)    


def tests():
    data_path = os.environ['PYSPTOOLS_DATA']
    project_path = os.environ['HOME']
    result_path = osp.join(project_path, 'results')
    sample = '92AV3C.hdr'

    data_file = osp.join(data_path, sample)
    hcube, header = util.load_ENVI_file(data_file)

    hcube_clean, header['wavelength'] = remove_bands(hcube)

    RB_amaps, RB_E = data_mine(result_path, hcube_clean)

    RB_m = mask(result_path, hcube_clean, RB_amaps)


    ecv = test_HyperEstimatorCrossVal(hcube_clean, RB_m)
    test_HyperLogisticRegression(result_path, hcube_clean, RB_m, ecv)

    test_HyperSVC(result_path, hcube_clean, RB_m)
    test_HyperRandomForestClassifier(result_path, hcube_clean, RB_m)
    test_HyperGradientBoostingClassifier(result_path, hcube_clean, RB_m)
    test_HyperKNeighborsClassifier(result_path, hcube_clean, RB_m)
    test_HyperAdaBoostClassifier(result_path, hcube_clean, RB_m)
    test_HyperExtraTreesClassifier(result_path, hcube_clean, RB_m)
    test_HyperBaggingClassifier(result_path, hcube_clean, RB_m)

    
# What does the tests:
# One third of the cube is use to train, the left part with a vertical split.
# After the training, all the cube is fit. The right two third is evaluated
# by visual inspection.
#
# Results
# By visual inpection and without parameters tuning, from best to worst:
#
# HyperGradientBoostingClassifier
# HyperKNeighborsClassifier
# HyperSVC
# HyperRandomForestClassifier
# HyperLogisticRegression
 

if __name__ == '__main__':
    import sys
    print(sys.version_info)
    tests()
