#

from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sklearn.model_selection as ms
import tabulate as tb

from .hyperxgb import HyperXGBClassifier
from .hyperlgbm import HyperLGBMClassifier

import xgboost as xgb
from xgboost.compat import XGBLabelEncoder

import lightgbm as lgbm
from lightgbm.compat import _LGBMLabelEncoder

# __pprint_dic
def _pprint(param_map, prefix='  '):
    list_key_value = [[k,str(v)] for k, v in sorted(param_map.items())]
    for k,v in list_key_value:
        print(prefix+'{:30} {}'.format(k,v))
    

class Tune(object):
    
    def __init__(self, estimator, p_ini, X, y):
        self.estimator = estimator
        self.p_current = p_ini
        self.X = X
        self.y = y

    def step_GradientBoostingCV(self, update=None, cv_params={}, verbose=False):
        if update != None:
            self.p_update(update)
        if self.estimator == HyperXGBClassifier:
            cv = XGBoostCV(self.p_current, **cv_params)
        if self.estimator == HyperLGBMClassifier:
            cv = LightGBMCV(self.p_current, **cv_params)
        cv.fit(self.X, self.y)
        if verbose == True:
            print('----------------------------------------------------------------')
            cv.print_cv_tail(); print()

    def step_GridSearchCV(self, p_grid, title=None, verbose=False):
        for k,v in p_grid.items():
            try:
                del self.p_current[k]
            except:
                pass
        cv = HyperGridSearchCV(self.estimator, self.p_current, p_grid)
        cv.fit(self.X, self.y)
        if verbose == True:
            cv.print(title)
        best = cv.get_best_params()
        self.p_current.update(best)

    def p_update(self, p_up):
        self.p_current.update(p_up)
#        print('p_update')
#        print(self.p_current)
        
    def get_p_current(self):
        return self.p_current
        
    def print_params(self, prefix=''):
        print('----------------------------------------------------------------')
        print('{} parameters:'.format(prefix))
        print()
        head = ['parameter','value']
        print(tb.tabulate(sorted(self.p_current.items()), headers=head, tablefmt="simple"))
        print()


class LightGBMCV(object):
    # estimator_param -> param_map
    def __init__(self, estimator_param, num_boost_round=10,
               folds=None, nfold=5, stratified=True, shuffle=True,
               metrics=None, fobj=None, feval=None, init_model=None,
               feature_name='auto', categorical_feature='auto',
               early_stopping_rounds=None, fpreproc=None,
               verbose_eval=None, show_stdv=True, seed=0,
               callbacks=None):
#        e = HyperLGBMClassifier(**estimator_param)
#        self.param_map = e.get_params()
        self.param_map = estimator_param
        self.num_boost_round=num_boost_round
        self.folds=folds
        self.nfold=nfold
        self.stratified=stratified
        self.shuffle=shuffle
        self.metrics=metrics
        self.fobj=fobj
        self.feval=feval
        self.init_model=init_model
        self.feature_name=feature_name
        self.categorical_feature=categorical_feature
        self.early_stopping_rounds=early_stopping_rounds
        self.fpreproc=fpreproc
        self.verbose_eval=verbose_eval
        self.show_stdv=show_stdv
        self.seed=seed
        self.callbacks=callbacks
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._le = _LGBMLabelEncoder().fit(y)
        training_labels = self._le.transform(y)
        xgdmat = lgbm.Dataset(X, label=training_labels)
        #xgdmat.construct() 
        self.param_map.update({'objective':'binary'})
        #print('avant lgbm.cv')
        #print(self.param_map)
        # a verifier
#        if self.n_classes_ > 2:
#            self.param_map.update({'num_class':self.n_classes_})
#            self.param_map.update({'objective':'multi:softprob'})
        # Note: lgbm.cv reset the value of max_bin to 255
        self.results = lgbm.cv(self.param_map,
                               xgdmat,
                                self.num_boost_round,
                                self.folds,
                                self.nfold,
                                self.stratified,
                                self.shuffle,
                                self.metrics,
                                self.fobj,
                                self.feval,
                                self.init_model,
                                self.feature_name,
                                self.categorical_feature,
                                self.early_stopping_rounds,
                                self.fpreproc,
                                self.verbose_eval,
                                self.show_stdv,
                                self.seed,
                                self.callbacks)
        #print('avant lgbm.cv')
        #print(self.param_map)
        
    def print_cv_tail(self):
        print('LightGBM cross-validation tail'); print()
        #print(self.results.tail(5))
        print(self.results)

    def print_param_map(self):
        print('----------------------------------------------------------------')
        print('LightGBMCV: parameters map')
        head = ['parameter','value']
        data = [[k,str(v)] for k, v in sorted(self.param_map.items())]
        print(tb.tabulate(data, headers=head, tablefmt="simple"))
#        for k,v in list_key_value:
#            print('{:30} {}'.format(k,v))

        
class XGBoostCV(object):
    # estimator_param -> param_map
    def __init__(self, estimator_param, num_boost_round=10, nfold=3, stratified=False, folds=None,
                 metrics=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None,
                 fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=0,
                 callbacks=None):
        # necessaire?
#        e = xgb.XGBClassifier(**estimator_param)
#        self.param_map = e.get_xgb_params()
        self.param_map = estimator_param
        #print(self.param_map)
        self.num_boost_round=num_boost_round
        self.nfold=nfold
        self.stratified=stratified
        self.folds=folds
        self.metrics=metrics
        self.obj=obj
        self.feval=feval
        self.maximize=maximize
        self.early_stopping_rounds=early_stopping_rounds
        self.fpreproc=fpreproc
        self.as_pandas=as_pandas
        self.verbose_eval=verbose_eval
        self.show_stdv=show_stdv
        self.seed=seed
        self.callbacks=callbacks        
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._le = XGBLabelEncoder().fit(y)
        training_labels = self._le.transform(y)
        xgdmat = xgb.DMatrix(X, label=training_labels)
        if self.n_classes_ > 2:
            self.param_map.update({'num_class':self.n_classes_})
            self.param_map.update({'objective':'multi:softprob'})
        self.results = xgb.cv(self.param_map,
                          xgdmat,
                          self.num_boost_round,
                          self.nfold,
                          self.stratified,
                          self.folds,
                          self.metrics,
                          self.obj,
                          self.feval,
                          self.maximize,
                          self.early_stopping_rounds,
                          self.fpreproc,
                          self.as_pandas,
                          self.verbose_eval,
                          self.show_stdv,
                          self.seed,
                          self.callbacks)
        
    def print_cv_tail(self):
        print('XGBoost cross-validation tail'); print()
        print(self.results.tail(5))

    def print_param_map(self):
        print('----------------------------------------------------------------')
        print('XGBoostCV: parameters map')
        head = ['parameter','value']
        data = [[k,str(v)] for k, v in sorted(self.param_map.items())]
        print(tb.tabulate(data, headers=head, tablefmt="simple"))
#        for k,v in list_key_value:
#            print('{:30} {}'.format(k,v))
        

class HyperGridSearchCV(object):
    """ Do a cross validation on a hypercube or on a concatenation of hypercubes.
        Use scikit-learn KFold and GridSearchCV. """

    def __init__(self, estimator, estimator_param, param_grid, n_splits=2):
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
        self.estimator_param = estimator_param
        self.param_grid = param_grid
        self.n_splits = n_splits

    def fit(self, X, y):
        """
        Run the cross validation.

        Parameters:
            X: `numpy array`
                A vector (n_samples, n_features) where each element *n_features* is a spectrum.

            y: `numpy array`
                Target values (n_samples,). A zero value is the background. A value of one or more is a class value.
        """
        self._cross_val(X, y)

    def _cross_val(self, X, Y):
        kf = ms.KFold(n_splits=self.n_splits, shuffle=True)
        self.gcv = ms.GridSearchCV(self.estimator(**self.estimator_param),
                                   self.param_grid,
                                   cv=kf,
                                   refit=False)
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
#        params = sorted(self.gcv.cv_results_['params'])
        params = self.gcv.cv_results_['params']
        scores = self.gcv.cv_results_['mean_test_score']
        stds = self.gcv.cv_results_['std_test_score']
        print('----------------------------------------------------------------')
        print('Cross validation inputs:'); print()
        print('n splits:', self.n_splits)
        print('Shuffle: True')
        print()
        print('Parameters grid:'); print()
        print(tb.tabulate(sorted(self.param_grid.items()), tablefmt="simple"))
        #_pprint(self.param_grid); print()
        print()
        print('----------------------------------------------------------------')
        print('Cross validation results:'); print()
        print('Best score:', self.gcv.best_score_); print()
        print('Best parameters:'); print()
        print(tb.tabulate(sorted(self.gcv.best_params_.items()), tablefmt="simple"))
        #_pprint(self.gcv.best_params_, prefix='    ')
        header = [k for k, v in sorted(self.gcv.best_params_.items())]+['score','std']
        print()
        #print('----------------------------------------------------------------')
        print('All scores:'); print()
        data = list()
        for p,sc,st in zip(params,scores,stds):
            value_list = [v for k, v in sorted(p.items())]
            value_list.append(sc)
            value_list.append(st)
            data.append(value_list)

        print(tb.tabulate(data, headers=header, tablefmt="simple"))
        print()
