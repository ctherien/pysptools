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
# naive_bayes.py - This file is part of the PySptools package.
#

import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base import HyperBaseClassifier


class HyperGaussianNB(GaussianNB, HyperBaseClassifier):
    """
    Apply scikit-learn GaussianNB on a hypercube.
    
    For the __init__ class contructor parameters: `see the sklearn.naive_bayes.GaussianNB class parameters`

    
    The class is intrumented to be use with the scikit-learn cross validation.
    It use the plot and display methods from the class Output.
    """

    def __init__(self, priors=None):
        super(HyperGaussianNB, self).__init__(priors=priors)
        HyperBaseClassifier.__init__(self, 'HyperGaussianNB')

    def fit(self, X, y, sample_weight=None):
        """
        Same as the sklearn.naive_bayes.GaussianNB fit call.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.
        """
        super(HyperGaussianNB, self)._set_n_clusters(int(np.max(y)))
        super(HyperGaussianNB, self).fit(X, y, sample_weight=sample_weight)

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
        super(HyperGaussianNB, self).fit(X, y)

    def classify(self, M):
        """
        Classify a hyperspectral cube.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

        Returns: `numpy array`
              A class map (m x n x 1).
        """
        img = self._convert2D(M)
        cls = super(HyperGaussianNB, self).predict(img)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperGaussianNB, self)._set_cmap(cmap)
        return self.cmap
