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
# inval.py - This file is part of the PySptools package.
#

"""
"""

import pysptools.util as util


# SAM, SID, NormXCorr
def ClassifyInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, E, threshold=0.1, mask=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.lib_type, method.__name__, E, 'E')
            check.dispatch(check.threshold_type, method.__name__, E, threshold)
            check.dispatch(check.spectrum_length, method.__name__, M, 'M', E, 'E')
            check.dispatch(check.mask_type, method.__name__, mask)
            return method(self, M, E, threshold=threshold, mask=mask)
        return checker
    return wrap


# AbundanceClassification
def ClassifyInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, threshold=0.1):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.threshold_type3, method.__name__, M, threshold)
            return method(self, M, threshold=threshold)
        return checker
    return wrap


# SVC
def ClassifyInputValidation3(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            return method(self, M)
        return checker
    return wrap


# KMeans
def PredictInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, n_clusters=5, n_jobs=1, init='k-means++'):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.int_type, method.__name__, n_clusters, 'n_clusters')
            check.dispatch(check.int_type, method.__name__, n_jobs, 'n_jobs')
            # init is string or array
            #check.dispatch(check.string_type, method.__name__, init, 'init')
            return method(self, M, n_clusters=n_clusters, n_jobs=n_jobs, init=init)
        return checker
    return wrap


# SAM, SID, NormXCorr
def GetMapInputValidation(class_name, call_before):
    @util.simple_decorator
    def wrap(method):
        def checker(self):
            check = util.InputValidation(class_name)
            check.dispatch(check.cmap_exist, method.__name__, self.cmap, call_before)
            return method(self)
        return checker
    return wrap


# SAM, SID, NormXCorr
def GetSingleMapInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, lib_idx, constrained=True):
            check = util.InputValidation(class_name)
            check.dispatch(check.index_range, method.__name__, lib_idx, self.n_classes)
            check.dispatch(check.bool_type, method.__name__, constrained, 'constrained')
            return method(self, lib_idx, constrained=constrained)
        return checker
    return wrap


# SAM, SID, NormXCorr
def PlotSingleMapInputValidation(class_name, call_before):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cmap_exist, method.__name__, self.cmap, call_before)
            check.dispatch(check.index_range, method.__name__, lib_idx, self.n_classes)
            check.dispatch(check.bool_type, method.__name__, constrained, 'constrained')
            check.dispatch(check.bool_type, method.__name__, stretch, 'stretch')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, lib_idx, constrained=constrained, stretch=stretch, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# SAM, SID, NormXCorr
def DisplaySingleMapInputValidation(class_name, call_before):
    @util.simple_decorator
    def wrap(method):
        def checker(self, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cmap_exist, method.__name__, self.cmap, call_before)
            check.dispatch(check.index_range, method.__name__, lib_idx, self.n_classes)
            check.dispatch(check.bool_type, method.__name__, constrained, 'constrained')
            check.dispatch(check.bool_type, method.__name__, stretch, 'stretch')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, lib_idx, constrained=constrained, stretch=stretch, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# SAM, SID, NormXCorr, AbundanceValidation
def PlotInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.labels_type, method.__name__, labels)
            check.dispatch(check.mask_type, method.__name__, mask)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, labels=labels, mask=mask, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# SAM, SID, NormXCorr, AbundanceValidation
def DisplayInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.labels_type, method.__name__, labels)
            check.dispatch(check.mask_type, method.__name__, mask)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, labels=labels, mask=mask, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# SVC
def PlotInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, labels=None, interpolation='none', colorMap='Accent', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.labels_type, method.__name__, labels)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, labels=labels, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# SVC
def DisplayInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, labels=None, interpolation='none', colorMap='Accent', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.labels_type, method.__name__, labels)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, labels=labels, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# Kmeans
def PlotInputValidation3(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, interpolation='none', colorMap='Accent', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# Kmeans
def DisplayInputValidation3(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, interpolation='none', colorMap='Accent', suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        return checker
    return wrap


# SAM, SID, NormXCorr
def PlotHistoInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, suffix=suffix)
        return checker
    return wrap
