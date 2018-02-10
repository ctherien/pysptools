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
# ensemble.py - This file is part of the PySptools package.
#


import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
                             AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from .base import HyperBaseClassifier
from .util import _plot_feature_importances
from .docstring import *


def _document(cls):
    import sys
    if sys.version_info[0] == 2:
        cls.plot_feature_importances.__func__.__doc__ = plot_fi_docstring
        cls.display_feature_importances.__func__.__doc__ = display_fi_docstring
    if sys.version_info[0] == 3:
        cls.plot_feature_importances.__doc__ = plot_fi_docstring
        cls.display_feature_importances.__doc__ = display_fi_docstring


class HyperAdaBoostClassifier(AdaBoostClassifier, HyperBaseClassifier):
    """
    Apply scikit-learn AdaBoostClassifier on a hypercube.
    
    For the __init__ class contructor parameters: `see the sklearn.ensemble.AdaBoostClassifier class parameters`

    The class is intrumented to be use with the scikit-learn cross validation.
    It use the plot and display methods from the class Output.
    """

    cmap = None

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):
        super(HyperAdaBoostClassifier, self).__init__(
                 base_estimator=base_estimator,
                 n_estimators=n_estimators,
                 learning_rate=learning_rate,
                 algorithm=algorithm,
                 random_state=random_state)
        HyperBaseClassifier.__init__(self, 'HyperAdaBoostClassifier')

    def fit(self, X, y, sample_weight=None):
        """
        Same as the sklearn.ensemble.GradientBoostingClassifier fit call.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.
        
            sample_weight : array-like of shape = [n_samples], optional
                Sample weights. If None, the sample weights are initialized to
                ``1 / n_samples``.

        """
        super(HyperAdaBoostClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperAdaBoostClassifier, self).fit(X, y, sample_weight)

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
        super(HyperAdaBoostClassifier, self).fit(X, y)

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
        cls = super(HyperAdaBoostClassifier, self).predict(img)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperAdaBoostClassifier, self)._set_cmap(cmap)
        return self.cmap

    def plot_feature_importances(self, path, n_labels='all',
                                 height=0.2, sort=False, suffix=None):
        _plot_feature_importances('HyperAdaBoostClassifier', self.feature_importances_, path,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

    def display_feature_importances(self, n_labels='all',
                                    height=0.2, sort=False, suffix=None):
        _plot_feature_importances('', self.feature_importances_, None,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

_document(HyperAdaBoostClassifier)

        
class HyperExtraTreesClassifier(ExtraTreesClassifier, HyperBaseClassifier):
    """
    Apply scikit-learn ExtraTreesClassifier on a hypercube.
    
    For the __init__ class contructor parameters: `see the sklearn.ensemble.ExtraTreesClassifier`

    The class is intrumented to be use with the scikit-learn cross validation.
    It use the plot and display methods from the class Output.
    """

    cmap = None

    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(HyperExtraTreesClassifier, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
        HyperBaseClassifier.__init__(self, 'HyperExtraTreesClassifier')

    def fit(self, X, y, sample_weight=None):
        """
        Same as the sklearn.ensemble.ExtraTreesClassifier fit call.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.

            sample_weight : array-like, shape = [n_samples] or None
                Sample weights. If None, then samples are equally weighted. Splits
                that would create child nodes with net zero or negative weight are
                ignored while searching for a split in each node. In the case of
                classification, splits are also ignored if they would result in any
                single class carrying a negative weight in either child node.
        """
        super(HyperExtraTreesClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperExtraTreesClassifier, self).fit(X, y, sample_weight)

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
        super(HyperExtraTreesClassifier, self).fit(X, y)

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
        cls = super(HyperExtraTreesClassifier, self).predict(img)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperExtraTreesClassifier, self)._set_cmap(cmap)
        return self.cmap

    def plot_feature_importances(self, path, n_labels='all',
                                 height=0.2, sort=False, suffix=None):
        _plot_feature_importances('HyperExtraTreesClassifier', self.feature_importances_, path,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

    def display_feature_importances(self, n_labels='all',
                                    height=0.2, sort=False, suffix=None):
        _plot_feature_importances('', self.feature_importances_, None,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

_document(HyperExtraTreesClassifier)

        
class HyperBaggingClassifier(BaggingClassifier, HyperBaseClassifier):
    """
    Apply scikit-learn BaggingClassifier on a hypercube.
    
    For the __init__ class contructor parameters: `see the sklearn.ensemble.BaggingClassifier class parameters`

    The class is intrumented to be use with the scikit-learn cross validation.
    It use the plot and display methods from the class Output.
    """

    cmap = None

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(HyperBaggingClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        HyperBaseClassifier.__init__(self, 'HyperBaggingClassifier')

    def fit(self, X, y, sample_weight=None):
        """
        Same as the sklearn.ensemble.BaggingClassifier fit call.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.

            sample_weight : array-like, shape = [n_samples] or None
                Sample weights. If None, then samples are equally weighted.
                Note that this is supported only if the base estimator supports
                sample weighting.
        """
        super(HyperBaggingClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperBaggingClassifier, self).fit(X, y, sample_weight)

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
        super(HyperBaggingClassifier, self).fit(X, y)

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
        cls = super(HyperBaggingClassifier, self).predict(img)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperBaggingClassifier, self)._set_cmap(cmap)
        return self.cmap

    
class HyperGradientBoostingClassifier(GradientBoostingClassifier, HyperBaseClassifier):
    """
    Apply scikit-learn GradientBoostingClassifier on a hypercube.
    
    For the __init__ class contructor parameters: `see the sklearn.ensemble.GradientBoostingClassifier class parameters`

    The class is intrumented to be use with the scikit-learn cross validation.
    It use the plot and display methods from the class Output.
    """

    cmap = None

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_split=1e-7, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto'):
        super(HyperGradientBoostingClassifier, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start,
            presort=presort)
        HyperBaseClassifier.__init__(self, 'HyperGradientBoostingClassifier')

    def fit(self, X, y):
        """
        Same as the sklearn.ensemble.GradientBoostingClassifier fit call.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.
        """
        super(HyperGradientBoostingClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperGradientBoostingClassifier, self).fit(X, y)

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
        super(HyperGradientBoostingClassifier, self).fit(X, y)

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
        cls = super(HyperGradientBoostingClassifier, self).predict(img)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperGradientBoostingClassifier, self)._set_cmap(cmap)
        return self.cmap

    def plot_feature_importances(self, path, n_labels='all',
                                 height=0.2, sort=False, suffix=None):
        _plot_feature_importances('HyperGBC', self.feature_importances_, path,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

    def display_feature_importances(self, n_labels='all',
                                    height=0.2, sort=False, suffix=None):
        _plot_feature_importances('', self.feature_importances_, None,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

_document(HyperGradientBoostingClassifier)


class HyperRandomForestClassifier(RandomForestClassifier, HyperBaseClassifier):
    """
    Apply scikit-learn RandomForestClassifier on a hypercube.
    
    For the __init__ class contructor parameters: `see the sklearn.ensemble.RandomForestClassifier class parameters`

    The class is intrumented to be use with the scikit-learn cross validation.
    It use the plot and display methods from the class Output.
    """

    cmap = None

    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(HyperRandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
        HyperBaseClassifier.__init__(self, 'HyperRandomForestClassifier')

    def fit(self, X, y):
        """
        Same as the sklearn.ensemble.RandomForestClassifier fit call.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.
        """
        super(HyperRandomForestClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperRandomForestClassifier, self).fit(X, y)

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
        super(HyperRandomForestClassifier, self).fit(X, y)

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
        cls = super(HyperRandomForestClassifier, self).predict(img)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperRandomForestClassifier, self)._set_cmap(cmap)
        return self.cmap

    def plot_feature_importances(self, path, n_labels='all',
                                 height=0.2, sort=False, suffix=None):
        _plot_feature_importances('HyperRF', self.feature_importances_, path,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

    def display_feature_importances(self, n_labels='all',
                                    height=0.2, sort=False, suffix=None):
        _plot_feature_importances('', self.feature_importances_, None,
                                  n_labels=n_labels, height=height, sort=sort,
                                  suffix=suffix)

_document(HyperRandomForestClassifier)
