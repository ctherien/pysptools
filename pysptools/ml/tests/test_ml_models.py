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
# test_ml_models.py - This file is part of the PySptools package.
#

from __future__ import print_function
import os
import os.path as osp
import numpy as np

import pysptools.ml as ml
import pysptools.skl as skl

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
    

def tune_xgb():
    return {'objective':"binary:logistic",'scale_pos_weight': 1.5, 'n_estimators': 823,
            'reg_alpha': 0.01, 'subsample': 0.6, 'reg_lambda': 1, 'min_child_weight': 1,
            'gamma': 0.3, 'max_depth': 26, 'learning_rate': 0.01, 'colsample_bytree': 0.6}


#def tune_lgbm():
#    return {'boosting_type':"gbdt", 'num_leaves':31, 'max_depth':-1,
#                 'learning_rate':0.1, 'n_estimators':10, 'max_bin':255,
#                 'subsample_for_bin':50000, 'objective':None,
#                 'min_split_gain':0., 'min_child_weight':5, 'min_child_samples':10}

def tune_lgbm():
    return {'boosting_type':"gbdt", 'num_leaves':10, 'max_depth':20,
            'learning_rate':0.1, 'n_estimators':10,
            'subsample_for_bin':50000, 'objective':None,
            'min_split_gain':0., 'min_child_weight':5, 'min_child_samples':10}


def fit_model(rpath, X_train, y_train, X_test, y_test, estimator, param, stat=False):
    model = estimator(**param)
    model.fit(X_train, y_train)
    if stat == True:
        accuracy(model, X_test, y_test)
    if rpath == None:
        model.display_feature_importances(height=0.6, sort=True, n_labels='all')
    else:
        model.plot_feature_importances(rpath, height=0.6, sort=True, n_labels='all')
    return model


def accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    ## evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def get_scaled_and_class_map(spath, rpath, fname, display):
    scl = []
    cmap = []
    
    # img1
    if 'img1' in fname:
        img1_scaled, img1_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'img1', 
                                  [['Snow',{'rec':(41,79,49,100)}]],
                                  skl.HyperGaussianNB, None,
                                  display=display)
        scl.append(img1_scaled)
        cmap.append(img1_cmap)
    
    # img2
    if 'img2' in fname:
        img2_scaled, img2_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'img2', 
                                  [['Snow',{'rec':(83,50,100,79)},{'rec':(107,151,111,164)}]],
                                  skl.HyperLogisticRegression, {'class_weight':{0:1.0,1:5}},
                                  display=display)    
        scl.append(img2_scaled)
        cmap.append(img2_cmap)
        
    # imgc7
    if 'imgc7' in fname:
        imgc7_scaled, imgc7_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'imgc7', 
                                  [['Snow',{'rec':(104,4,126,34)},{'rec':(111,79,124,101)}]],
                                  skl.HyperSVC, {'class_weight':{0:1,1:10},'gamma':0.5},
                                  display=display)
        # Clean the top half:
        imgc7_cmap[0:50,0:imgc7_cmap.shape[1]] = 0
        ml.display_img(imgc7_cmap, 'imgc7 class map cleaned')
        scl.append(imgc7_scaled)
        cmap.append(imgc7_cmap)
        
    # imga6
    if 'imga6' in fname:
        imga6_scaled, imga6_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'imga6', 
                                  [['Snow',{'rec':(5,134,8,144)}]],
                                  skl.HyperLogisticRegression, {'class_weight':{0:1.0,1:5}},
                                  display=display)
        scl.append(imga6_scaled)
        cmap.append(imga6_cmap)
            
    # imgb3
    if 'imgb3' in fname:
        imgb3_scaled, imgb3_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'imgb3', 
                                  [['Snow',{'rec':(99,69,103,95)}]],
                                  skl.HyperLogisticRegression, {'class_weight':{0:1.0,1:5}},
                                  display=display)
        scl.append(imgb3_scaled)
        cmap.append(imgb3_cmap)
        
    # imgc1
    if 'imgc1' in fname:
        imgc1_scaled, imgc1_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'imgc1', 
                                  [['Snow',{'rec':(51,69,54,91)},{'rec':(101,3,109,16)}]],
                                  skl.HyperLogisticRegression, {'class_weight':{0:1.0,1:5}},
                                  display=display)
        scl.append(imgc1_scaled)
        cmap.append(imgc1_cmap)
        
    # imgc4
    if 'imgc4' in fname:
        imgc4_scaled, imgc4_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'imgc4', 
                                  [['Snow',{'rec':(47,61,49,63)}]],
                                  skl.HyperSVC, {'class_weight':{0:0.05,1:40}},
                                  display=display)
        scl.append(imgc4_scaled)
        cmap.append(imgc4_cmap)
            
    # imgc5
    if 'imgc5' in fname:
        imgc5_scaled, imgc5_cmap = ml.get_scaled_img_and_class_map(spath, rpath, 'imgc5', 
                                  [['Snow',{'rec':(17,151,20,156)}]],
                                  skl.HyperLogisticRegression, {'class_weight':{0:1.0,1:5}},
                                  display=display)
        scl.append(imgc5_scaled)
        cmap.append(imgc5_cmap)
        
    return scl, cmap


def learn_and_classify(estimator, tune, label, save_model=False):
    home_path = os.environ['HOME']
    source_path = osp.join(home_path, 'dev-data/CZ_hsdb')
    result_path = osp.join(home_path, 'results')
    
    print('###########################################')
    print('Learn and classify for ', label)
    n_shrink = 3
    
    snow_fname1 = ['img1','img2']
    snow_fname2 = ['img1','img2','imgc7','imga6','imgb3','imgc1','imgc4','imgc5']
    nosnow_fname = ['imga1','imgb1','imgb6','imga7']
    
    snow_img, snow_cmap = get_scaled_and_class_map(source_path, result_path, snow_fname1, True)
    nosnow_img = ml.batch_load(source_path, nosnow_fname, n_shrink)
        
    M = snow_img[0]
    bkg_cmap = np.zeros((M.shape[0],M.shape[1]))
        
    X,y = skl.shape_to_XY(snow_img+nosnow_img, 
                          snow_cmap+[bkg_cmap,bkg_cmap,bkg_cmap,bkg_cmap])
    seed = 5
    train_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                        random_state=seed)
    best_param = tune()
    
    model = fit_model(result_path, X_train, y_train, X_test, y_test, 
                      estimator, best_param, stat=True)
    
    if save_model == True:
        if estimator == ml.HyperLGBMClassifier:
            model.save(osp.join(result_path, 'lgbm_model'), X_train.shape[1], 2)
        if estimator == ml.HyperXGBClassifier:
            model.save(osp.join(result_path, 'xgb_model'), X_train.shape[1], 2)
    
        if estimator == ml.HyperLGBMClassifier:
            model = ml.load_lgbm_model(osp.join(result_path, 'lgbm_model'))
        if estimator == ml.HyperXGBClassifier:
            model = ml.load_xgb_model(osp.join(result_path, 'xgb_model'))
    
    ml.batch_classify(source_path, result_path, model, snow_fname2 + nosnow_fname, n_shrink)


def tests():
    learn_and_classify(ml.HyperLGBMClassifier, tune_lgbm, 'HyperLGBMClassifier', save_model=True)
    learn_and_classify(ml.HyperXGBClassifier, tune_xgb, 'HyperXGBClassifier', save_model=True)


if __name__ == '__main__':
    tests()
