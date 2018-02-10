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
# svm.py - This file is part of the PySptools package.
#


import numpy as np
from sklearn import preprocessing
from sklearn import svm
from .base import HyperScaledBaseClassifier


class HyperSVC(svm.SVC, HyperScaledBaseClassifier):
    """
    Apply scikit-learn SVC on a hypercube.
    
    For the __init__ class contructor parameters: `see the sklearn.svm.SVC class parameters`

    
    The class is intrumented to be use with the scikit-learn cross validation.
    It use the plot and display methods from the class Output.

    Note: the class always do a preprocessing.scale before any processing.
    
    Note: the C parameter is set to 1, the result of this setting is that
    the class_weight is relative to C and that the first value of
    class_weight is the background.
    An example: you wish to fit two classes "1" and "2" with the help
    of one ROI for each, you declare class_weight like this:
            * class_weight={0:1,1:10,2:10}
            * 0: is always the background and is set to 1, 1: is the first class,
            * 2: is the second. A value of 10 for both classes give good results to
              start with.
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape=None,
                 random_state=None):
        super(HyperSVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                 coef0=coef0, shrinking=shrinking, probability=probability,
                 tol=tol, cache_size=cache_size, class_weight=class_weight,
                 verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                 random_state=random_state)
        HyperScaledBaseClassifier.__init__(self, 'HyperSVC')

    def fit(self, X, y):
        """
        Same as the sklearn.svm.SVC fit call, but with preprocessing.scale call first.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.
        """
        super(HyperSVC, self)._set_n_clusters(int(np.max(y)))
        X_scaled = preprocessing.scale(X)
        super(HyperSVC, self).fit(X_scaled, y)

    def fit_rois(self, M, ROIs):
        """
        Fit the HS cube M with the use of ROIs.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

            ROIs: `ROIs type`
                Regions of interest instance.
        """
        X, y = self._fit_rois(M, ROIs)
        super(HyperSVC, self).fit(X, y)

    def classify(self, M):
        """
        Classify a hyperspectral cube. Do a preprocessing.scale before.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

        Returns: `numpy array`
              A class map (m x n x 1).
        """
        img = self._convert2D(M)
        image_scaled = preprocessing.scale(img)
        cls = super(HyperSVC, self).predict(image_scaled)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperSVC, self)._set_cmap(cmap)
        return cmap

    def predict(self, X):
        """
        Same as the sklearn.svm.SVC predict call, but with a call to preprocessing.scale first.

        Parameters:
            X: `numpy array`
                A vector where each element is a spectrum.
        """
        X_scaled = preprocessing.scale(X)
        y = super(HyperSVC, self).predict(X_scaled)
        return self.classes_.take(np.asarray(y, dtype=np.intp))

#    def fit_multi(self, M_list, mask_list):
#        """
#        Do a fit on a hypercube list where the sections
#        are determined by a list of binary mask. Only one cluster can be fit.
#        
#        PARTIALLY TESTED AND WILL CHANGE
#    
#        Parameters:
#            M_list: `numpy array list`
#                A list of HSI cube (m x n x p).
#    
#            mask_list: `numpy array list`
#                A list of binary mask, when *True* the corresponding spectrum is part of the
#                cross validation.        
#        """
#        self.n_clusters = 1
#        i = 0
#        for m,msk in zip(M_list, mask_list):
#            x = self._convert2D(m)
#            y = np.reshape(msk, msk.shape[0]*msk.shape[1])
#            if i == 0:
#                X = x
#                Y = y
#                i = 1
#            else:
#                X = np.concatenate((X, x))
#                Y = np.concatenate((Y, y))
#        X_scaled = preprocessing.scale(X)
#        super(HyperSVC, self).fit(X_scaled, Y)
        