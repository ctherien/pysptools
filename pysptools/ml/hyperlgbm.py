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
# hyperlgbm.py - This file is part of the PySptools package.
#

import numpy as np
import pickle

from pysptools.skl.base import HyperBaseClassifier
from pysptools.skl import _plot_feature_importances
from lightgbm import LGBMClassifier


# Hacks needed to run load_lgbm_model()
# Add to LGBMModel class (file sklearn.py):
#    # Patch pysptools
#    def set_n_features_(self, n):
#        self._n_features = n
#    # end patch pysptools
#
# Add to LGBMClassifier class:
#    # Patch pysptools
#    def set_le(self, y):
#        self._le = _LGBMLabelEncoder().fit(y)
#        self._classes = self._le.classes_
#        self._n_classes = len(self._classes)
#
#    def set_n_features_(self, n):
#        super(LGBMClassifier, self).set_n_features_(n)
#    # end patch pysptools


def load_lgbm_model(fname):
    """ Load a LightGBM model that was saved as a file with
        the HyperLGBMClassifier.save method.
        
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
    from lightgbm import Booster
    params = pickle.load(open(fname+'.p', "rb"))
    n_features = params['meta']['n_features']
    n_classes = params['meta']['n_classes']
    param_map = params['param_map']
    model = HyperLGBMClassifier(**param_map)
    model.set_n_labels(n_classes-1)
    y = [i for i in range(n_classes)]
    model.set_le(y)
    model.set_n_features_(n_features)
    model._Booster = Booster(model_file=fname)
    return model

  
class HyperLGBMClassifier(LGBMClassifier, HyperBaseClassifier):
    """
    LightGBM classifier for Hyperspectral Imaging.
    The class implement the scikit-learn API and is a pysptools submodule.
    
    This class add the save and load model functionalities.
    Following is a copy and paste form XGBModel documentation.

    Construct a gradient boosting model.

        Parameters
        ----------
        Parameters
        ----------
        boosting_type : string, optional (default="gbdt")
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'goss', Gradient-based One-Side Sampling.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, -1 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=50000)
            Number of samples for constructing bins.
        objective : string, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note that these weights will be multiplied with ``sample_weight`` (passed through the fit method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight(hessian) needed in a child(leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data need in a child(leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=1)
            Frequence of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int or None, optional (default=None)
            Random number seed.
            Will use default seeds in c++ code if set to None.
        n_jobs : int, optional (default=-1)
            Number of parallel threads.
        silent : bool, optional (default=True)
            Whether to print messages while running boosting.

        Attributes
        ----------
        n_features_ : int
            The number of features of fitted model.
        classes_ : array of shape = [n_classes]
            The class label array (only for classification problem).
        n_classes_ : int
            The number of classes (only for classification problem).
        best_score_ : dict or None
            The best score of fitted model.
        best_iteration_ : int or None
            The best iteration of fitted model if ``early_stopping_rounds`` has been specified.
        objective_ : string or callable
            The concrete objective used while fitting this model.
        booster_ : Booster
            The underlying Booster of this model.
        evals_result_ : dict or None
            The evaluation results if ``early_stopping_rounds`` has been specified.
        feature_importances_ : array of shape = [n_features]
            The feature importances (the higher, the more important the feature).

        Note
        ----
        A custom objective function can be provided for the ``objective``
        parameter. In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess`` or
        ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true: array-like of shape = [n_samples]
                The target values.
            y_pred: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
            group: array-like
                Group/query data, used for ranking task.
            grad: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the gradient for each sample point.
            hess: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the second derivative for each sample point.

        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
        and you should group grad and hess in this way as well.
        """
    
    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=1, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True):
        super(HyperLGBMClassifier, self).__init__(boosting_type, num_leaves, max_depth,
                 learning_rate, n_estimators,
                 subsample_for_bin, objective, class_weight,
                 min_split_gain, min_child_weight, min_child_samples,
                 subsample, subsample_freq, colsample_bytree,
                 reg_alpha, reg_lambda, random_state,
                 n_jobs, silent)
        HyperBaseClassifier.__init__(self, 'HyperLGBMClassifier')
 
    def fit(self, X, y, 
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_metric="logloss",
            early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto', callbacks=None):
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Input feature matrix.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).
        sample_weight : array-like of shape = [n_samples] or None, optional (default=None)
            Weights of training data.
        init_score : array-like of shape = [n_samples] or None, optional (default=None)
            Init score of training data.
        group : array-like of shape = [n_samples] or None, optional (default=None)
            Group data of training data.
        eval_set : list or None, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation sets for early-stopping.
        eval_names : list of strings or None, optional (default=None)
            Names of eval_set.
        eval_sample_weight : list of arrays or None, optional (default=None)
            Weights of eval data.
        eval_class_weight : list or None, optional (default=None)
            Class weights of eval data.
        eval_init_score : list of arrays or None, optional (default=None)
            Init score of eval data.
        eval_group : list of arrays or None, optional (default=None)
            Group data of eval data.
        eval_metric : string, list of strings, callable or None, optional (default=None)
            If string, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note for more details.
        early_stopping_rounds : int or None, optional (default=None)
            Activates early stopping. The model will train until the validation score stops improving.
            Validation error needs to decrease at least every ``early_stopping_rounds`` round(s)
            to continue training.
        verbose : bool, optional (default=True)
            If True and an evaluation set is used, writes the evaluation progress.
        feature_name : list of strings or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of strings or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas categorical columns are used.
        callbacks : list of callback functions or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.

        Returns
        -------
        self : object
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following functions:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``.
        Returns (eval_name, eval_result, is_bigger_better) or
        list of (eval_name, eval_result, is_bigger_better)

            y_true: array-like of shape = [n_samples]
                The target values.
            y_pred: array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class)
                The predicted values.
            weight: array-like of shape = [n_samples]
                The weight of samples.
            group: array-like
                Group/query data, used for ranking task.
            eval_name: str
                The name of evaluation.
            eval_result: float
                The eval result.
            is_bigger_better: bool
                Is eval result bigger better, e.g. AUC is bigger_better.

        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].
        """
        super(HyperLGBMClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperLGBMClassifier, self).fit(X=X, y=y,
             sample_weight=sample_weight, init_score=init_score,
             eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight,
             eval_init_score=eval_init_score, eval_metric=eval_metric,
             early_stopping_rounds=None, verbose=True,
             feature_name=feature_name, categorical_feature=categorical_feature,
             callbacks=callbacks)

    def partial_fit(self, X, y,
            sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_metric="logloss",
            early_stopping_rounds=None, verbose=True,
            feature_name='auto', categorical_feature='auto', callbacks=None):
        """ See fit() method doc """
        super(HyperLGBMClassifier, self)._set_n_clusters(int(np.max(y)))
        super(HyperLGBMClassifier, self).partial_fit(X=X, y=y,
             sample_weight=sample_weight, init_score=init_score,
             eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight,
             eval_init_score=eval_init_score, eval_metric=eval_metric,
             early_stopping_rounds=None, verbose=True,
             feature_name=feature_name, categorical_feature=categorical_feature,
             callbacks=callbacks)
    
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
        super(HyperLGBMClassifier, self).fit(X, y)

    def set_n_labels(self, n):
        # hack for save and load functionalities
        super(HyperLGBMClassifier, self)._set_n_clusters(n)
        
    def set_le(self, y):
        # hack for save and load functionalities
        super(HyperLGBMClassifier, self).set_le(y)
        
    def set_n_features_(self, n):
        # hack for save and load functionalities
        super(HyperLGBMClassifier, self).set_n_features_(n)
        
    def classify(self, M, raw_score=False, num_iteration=0):
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
        #from sklearn.preprocessing import LabelEncoder
        img = self._convert2D(M)
        cls = super(HyperLGBMClassifier, self).predict(img,
                         raw_score=raw_score, num_iteration=num_iteration)
        cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        super(HyperLGBMClassifier, self)._set_cmap(cmap)
        return self.cmap

    def save(self, fname, n_features, n_classes):
        """
        Save the model and is parameters in two files.
        When the model is loaded, it instantiate an object of class
        HyperLGBMClassifier. See load_lgbm_model function doc.
        
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
        param_map = self.get_params()
        params = {'meta':meta, 'param_map': param_map}
        pickle.dump( params, open(fname+'.p', "wb" ))
        self.booster_.save_model(fname)

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
        _plot_feature_importances('HyperLGBM', self.feature_importances_, path, 
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
