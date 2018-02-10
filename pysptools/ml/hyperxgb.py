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
# hyperxgb.py - This file is part of the PySptools package.
#


# Use with xgboost version 0.6a2

import numpy as np
import pickle

from pysptools.skl.base import HyperBaseClassifier
from pysptools.skl import _plot_feature_importances
from xgboost import XGBClassifier


# TODO:
# partial_fit
# Notes:
# seed pas utilise, utilise par mknfold et cv

# Hacks needed to run load_xgb_model()
# Add to XGBClassifier class (file sklearn.py):
#    # Patch pysptools
#    def set_le(self, y):
#        self._le = XGBLabelEncoder().fit(y)
#    # end patch pysptools

def load_xgb_model(fname):
    """ Load a XGBoost model that was saved as a file with
        the HyperXGBClassifier.save method.
        
        The model is span on two files:
            
            * The first file contains the model saved with the Booster class,
            this file have no extension.
            
            * The second file contains the parameters used to create the model,
            this file have the extension '.p'.
            
        Parameters
        ----------
        fname : path 
                The file name without extension.
        """
    from xgboost import Booster
    params = pickle.load(open(fname+'.p', "rb"))
    n_classes = params['meta']['n_classes']
    param_map = params['param_map']
    model = HyperXGBClassifier(**param_map)
    model.set_n_labels(n_classes-1)
    y = [i for i in range(n_classes)]
    model.set_le(y)
    model._Booster = Booster(model_file=fname)
    return model

  
class HyperXGBClassifier(XGBClassifier, HyperBaseClassifier):
    """
    XGBoost classifier for Hyperspectral Imaging.
    The class implement the scikit-learn API and is a pysptools submodule.
    
    This class add the save and load model functionalities.
    
    Following is a copy and paste form XGBModel documentation.

    Implementation of the Scikit-Learn API for XGBoost.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth for base learners.
    learning_rate : float
        Boosting learning rate (xgb's "eta")
    n_estimators : int
        Number of boosted trees to fit.
    silent : boolean
        Whether to print messages while running boosting.
    objective : string or callable
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
    nthread : int
        Number of parallel threads used to run xgboost.
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : int
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float
        Subsample ratio of the training instance.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float
        Subsample ratio of columns for each split, in each level.
    reg_alpha : float (xgb's alpha)
        L1 regularization term on weights
    reg_lambda : float (xgb's lambda)
        L2 regularization term on weights
    scale_pos_weight : float
        Balancing of positive and negative weights.

    base_score:
        The initial prediction score of all instances, global bias.
    seed : int
        Random number seed.
    missing : float, optional
        Value in the data which needs to be present as a missing value. If
        None, defaults to np.nan.

    Note
    ----
    A custom objective function can be provided for the ``objective``
    parameter. In this case, it should have the signature
    ``objective(y_true, y_pred) -> grad, hess``:

    y_true: array_like of shape [n_samples]
        The target values
    y_pred: array_like of shape [n_samples]
        The predicted values

    grad: array_like of shape [n_samples]
        The value of the gradient for each sample point.
    hess: array_like of shape [n_samples]
        The value of the second derivative for each sample point
    """
    
    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None):
        super(HyperXGBClassifier, self).__init__(max_depth, learning_rate,
                                            n_estimators, silent, objective,
                                            nthread, gamma, min_child_weight,
                                            max_delta_step, subsample,
                                            colsample_bytree, colsample_bylevel,
                                            reg_alpha, reg_lambda,
                                            scale_pos_weight, base_score, seed, missing)
        HyperBaseClassifier.__init__(self, 'HyperXGBClassifier')
    
    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        """
        Fit gradient boosting classifier

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            Weight for each instance
        eval_set : list, optional
            A list of (X, y) pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int, optional
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        """
        super(HyperXGBClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperXGBClassifier, self).fit(X=X, y=y, sample_weight=sample_weight,
            eval_set=eval_set, eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose)

    def partial_fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        """ See fit() method doc """
        super(HyperXGBClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperXGBClassifier, self).partial_fit(X=X, y=y, sample_weight=sample_weight,
            eval_set=eval_set, eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    
    def fit_rois(self, M, ROIs):
        """
        Fit the HS cube M with the use of ROIs.

        Parameters
        ----------      
        M : numpy array
            A HSI cube (m x n x p).

        ROIs : ROIs class type
               Regions of interest instance.
        """
        X, y = self._fit_rois(M, ROIs)
        super(HyperXGBClassifier, self).fit(X, y)

    def set_n_labels(self, n):
        # hack for save and load functionalities
        super(HyperXGBClassifier, self)._set_n_clusters(n)

    def set_le(self, y):
        # hack for save and load functionalities
        super(HyperXGBClassifier, self).set_le(y)
        
    def classify(self, M, output_margin=False, ntree_limit=0):
        """
        Classify a hyperspectral cube.

        Parameters
        ----------   
        M : numpy array
            A HSI cube (m x n x p).

        Returns
        -------
        numpy array : a class map (m x n x 1)
        """
        img = self._convert2D(M)
        cls = super(HyperXGBClassifier, self).predict(img,
                        output_margin=output_margin, ntree_limit=ntree_limit)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperXGBClassifier, self)._set_cmap(cmap)
        return self.cmap

    def save(self, fname, n_features, n_classes):
        """
        Save the model and is parameters in two files.
        When the model is loaded, it instantiate an object of class
        HyperXGBClassifier. See load_xgb_model function doc.
        
        Parameters
        ----------
        fname : path
                The model file name.
        n_features : int
                     The model number of features.
        n_classes : int
                    The model number of classes, ex. for a binary model
                    n_classes = 2 (the background is a class for pysptools).
        """
        meta = {'n_features':n_features, 'n_classes':n_classes}
        param_map = self.get_xgb_params()
        params = {'meta':meta, 'param_map': param_map}
        pickle.dump( params, open(fname+'.p', "wb" ))
        self.booster().save_model(fname)

    def plot_feature_importances(self, path, n_labels='all', height=0.2, sort=False, suffix=''):
        """
        Plot the feature importances.
        The output can be split in n graphs.

        Parameters
        ----------
        path : string
          The path where to save the plot.

        n_labels : string or integer
          The number of labels to output by graph. If the value is 'all',
          only one graph is generated.

        height : float [default 0.2]
          The bar height (in fact width).

        sort : boolean [default False]
          If true the feature importances are sorted.

        suffix : string [default None]
          Add a suffix to the file name.
        """
        _plot_feature_importances('HyperXGBC', self.feature_importances_, path, 
                                  n_labels=n_labels, height=height, sort=sort, suffix=suffix)

    def display_feature_importances(self, n_labels='all', height=0.2, sort=False, suffix=''):
        """
        Display the feature importances.
        The output can be split in n graphs.

        Parameters
        ----------
        n_labels : string or integer
          The number of labels to output by graph. If the value is 'all',
          only one graph is generated.

        height : float [default 0.2]
          The bar height (in fact width).

        sort : boolean [default False]
          If true the feature importances are sorted.

        suffix : string [default None]
          Add a suffix to the file name.
        """
        _plot_feature_importances('', self.feature_importances_, None, 
                                  n_labels=n_labels, height=height, sort=sort, suffix=suffix)
