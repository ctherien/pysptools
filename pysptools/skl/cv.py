#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2017, Christian Therien
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
# cv.py - This file is part of the PySptools package.
#

from __future__ import print_function

import numpy as np
import sklearn.model_selection as ms


class HyperEstimatorCrossVal(object):
    """ Do a cross validation on a hypercube or a concatenation of hypercubes.
        Use scikit-learn KFold and GridSearchCV. """

    def __init__(self, estimator, param_grid):
        """
        Create a new HyperEstimatorCrossVal.

        Parameters:
            estimator: `class name`
                One of HyperSVC, HyperRandomForestClassifier, HyperKNeighborsClassifier
                HyperLogisticRegression, HyperGradientBoostingClassifier.

            param_grid: `dic`
                A dic of parameters to be cross validated.
                Ex. for HyperSVC: {'C': [10,20,30,50], 'gamma': [0.1,0.5,1.0,10.0]}.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_splits = 2

#    Usefull ??
    def fit_cube(self, M, mask):
        """
        Do a cross validation on a hypercube
            
        Parameters:
            M: `numpy array`
                A HSI cube (m x n x p).

            mask: `numpy array`
                A class map mask.        
        """
        X = self._convert2D(M)
        Y = np.reshape(mask, mask.shape[0]*mask.shape[1])
        self._cross_val(X, Y, self.param_grid)

    def fit(self, X, y):
        """
        Run the cross validation.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.
        """
        self._cross_val(X, y, self.param_grid)

    def _cross_val(self, X, Y, grid, n_splits=2):
        self.n_splits = n_splits
        kf = ms.KFold(n_splits=n_splits, shuffle=True)
        self.gcv = ms.GridSearchCV(self.estimator(), grid, cv=kf, refit=False)
        self.gcv.fit(X, Y)

    def _convert2D(self, M):
        h, w, numBands = M.shape
        return np.reshape(M, (w*h, numBands))
        
    def get_best_params(self):
        """
        Returns: `dic`
            Dic of best match.
        """
        return self.gcv.best_params_
        
    def print(self, label='No title'):
        """
        Print a summary for the cross validation results.
        
        Parameters:
            label: `string`
                The test title.
        """
        params = self.gcv.cv_results_['params']
        scores = self.gcv.cv_results_['mean_test_score']
        stds = self.gcv.cv_results_['std_test_score']
        print('================================================================')
        print('Cross validation results for: {}'.format(label))
        print('Param grid:', self.param_grid)
        print('n splits:', self.n_splits)
        print('Shuffle: True')
        print('================================================================')
        print('Best score:', self.gcv.best_score_)
        print('Best params:', self.gcv.best_params_)
        print('================================================================')
        print('All scores')
        for p,sc,st in zip(params,scores,stds):
            print(p, ', score: '+str(sc), ', std: '+str(st))
        print('================================================================')
        print()
