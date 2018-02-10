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
# test_ml_cv.py - This file is part of the PySptools package.
#

from __future__ import print_function
import os
import os.path as osp
import numpy as np

import pysptools.ml as ml
import pysptools.skl as skl

from sklearn.model_selection import train_test_split
 

def print_step_header(step_id, title):
    print('================================================================')
    print('{}: {}'.format(step_id, title))
    print('================================================================')
    print()


def get_scaled_and_class_map(spath, fname, display):
    scl = []
    cmap = []
    
    # img1
    if 'img1' in fname:
        img1_scaled, img1_cmap = ml.get_scaled_img_and_class_map(spath, 'img1', 
                                  [['Snow',{'rec':(41,79,49,100)}]],
                                  skl.HyperGaussianNB, None,
                                  display=display)
        scl.append(img1_scaled)
        cmap.append(img1_cmap)
    
    # img2
    if 'img2' in fname:
        img2_scaled, img2_cmap = ml.get_scaled_img_and_class_map(spath, 'img2', 
                                  [['Snow',{'rec':(83,50,100,79)},{'rec':(107,151,111,164)}]],
                                  skl.HyperLogisticRegression, {'class_weight':{0:1.0,1:5}},
                                  display=display)    
        scl.append(img2_scaled)
        cmap.append(img2_cmap)
                
    return scl, cmap


def step_GradientBoostingCV(tune, update, cv_params, verbose):
    print_step_header('Step', 'GradientBoosting cross validation')
    tune.print_params('input')
    tune.step_GradientBoostingCV(update, cv_params, verbose)


def step_GridSearchCV(tune, params, title, verbose):
    print_step_header('Step', 'scikit-learn cross-validation')
    tune.print_params('input')
    tune.step_GridSearchCV(params, title, verbose)
    tune.print_params('output')


# Problems with:
# n_estimators : give a run time error inside the .dll
# max_bin is reseted to 255
start_param_lgbm = {'boosting_type':"gbdt",
                    'num_leaves':10,
                    'max_depth':-1,
                    'learning_rate':0.1,
                    #'n_estimators':10,
                    'max_bin':255,
                    'subsample_for_bin':50000,
                    'objective':None,
                    'min_split_gain':0.,
                    'min_child_weight':5,
                    'min_child_samples':10,
                    'subsample':1.,
                    'subsample_freq':1,
                    'colsample_bytree':1.,
                    'reg_alpha':0.,
                    'reg_lambda':0.}


def test_HyperLGBMClassifier_cv(X, y, start_param):
    print('################################################################')
    print('Cross Validation Tests for HyperLGBMClassifier')
    print('################################################################')
    print()
    
    t = ml.Tune(ml.HyperLGBMClassifier, start_param, X, y)
    
    # Step GradientBoostingCV
    # Note: give no results
#    step_GradientBoostingCV(t, {'max_depth':30}, {'verbose_eval':False}, True)
#    t.p_update({'n_estimators':184})
#    t.print_params('output')
    
    # Step GridSearchCV
    step_GridSearchCV(t, {'max_depth':[5,15,30,50], 'min_child_weight':[1]}, 'Step 2', True)


start_param_xgb = {'max_depth':10,
               'min_child_weight':1,
               'gamma':0,
               'subsample':0.8,
               'colsample_bytree':0.5,
               'scale_pos_weight':1.5}


def test_HyperXGBClassifier_cv(X, y, start_param):
    print('################################################################')
    print('Cross Validation Tests for HyperXGBClassifier')
    print('################################################################')
    print()
    
    t = ml.Tune(ml.HyperXGBClassifier, start_param, X, y)
    
    # Step GradientBoostingCV
    step_GradientBoostingCV(t, {'learning_rate':0.2,'n_estimators':5000,'silent':1},
                            {'verbose_eval':False},
                            True)
    t.p_update({'n_estimators':184})
    t.print_params('output')
    
    # Step GridSearchCV
    step_GridSearchCV(t, {'max_depth':[5,15,30,50], 'min_child_weight':[1]}, 'Step 2', True)

    
def init_cv():
    home_path = os.environ['HOME']
    source_path = osp.join(home_path, 'dev-data/CZ_hsdb')
    
    n_shrink = 3
    
    snow_fname1 = ['img1','img2']
    nosnow_fname = ['imga1','imgb1','imgb6','imga7']
    
    snow_img, snow_cmap = get_scaled_and_class_map(source_path, snow_fname1, False)
    nosnow_img = ml.batch_load(source_path, nosnow_fname, n_shrink)
        
    M = snow_img[0]
    bkg_cmap = np.zeros((M.shape[0],M.shape[1]))
        
    X,y = skl.shape_to_XY(snow_img+nosnow_img, 
                          snow_cmap+[bkg_cmap,bkg_cmap,bkg_cmap,bkg_cmap])
    seed = 5
    train_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                        random_state=seed)
    return X_train, y_train

    
def tests():
    X,y = init_cv()
    #test_HyperXGBClassifier_cv(X, y, start_param_xgb)
    test_HyperLGBMClassifier_cv(X, y, start_param_lgbm)


if __name__ == '__main__':
    tests()
