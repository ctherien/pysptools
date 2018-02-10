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


# SpectrumConvexHullQuotient
def SCHQInitInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, spectrum, wvl, normalize=False):
            check = util.InputValidation(class_name)
            check.dispatch(check.list_type, method.__name__, spectrum, 'spectrum')
            check.dispatch(check.list_type, method.__name__, wvl, 'wvl')
            check.dispatch(check.bool_type, method.__name__, normalize, 'normalize')
            return method(self, spectrum, wvl, normalize=normalize)
        return checker
    return wrap


# SpectrumConvexHullQuotient
def PlotInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, plot_name, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.string_type, method.__name__, plot_name, 'plot_name')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, plot_name, suffix=suffix)
        return checker
    return wrap


# SpectrumConvexHullQuotient
def DisplayInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, plot_name, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.string_type, method.__name__, plot_name, 'plot_name')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, plot_name, suffix=suffix)
        return checker
    return wrap


# FeaturesConvexHullQuotient
def FCHQInitInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, spectrum, wvl, startContinuum=None, stopContinuum=None, baseline=0, normalize=False):
            check = util.InputValidation(class_name)
            check.dispatch(check.list_type, method.__name__, spectrum, 'spectrum')
            check.dispatch(check.list_type, method.__name__, wvl, 'wvl')
            check.dispatch(check.float_type2, method.__name__, startContinuum, 'startContinuum')
            check.dispatch(check.float_type2, method.__name__, stopContinuum, 'stopContinuum')
            check.dispatch(check.float_type, method.__name__, baseline, 'baseline')
            check.dispatch(check.bool_type, method.__name__, normalize, 'normalize')
            return method(self, spectrum, wvl, startContinuum=startContinuum, stopContinuum=stopContinuum, baseline=baseline, normalize=normalize)
        return checker
    return wrap


# FeaturesConvexHullQuotient
def PlotInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, plot_name, feature='all'):
            check = util.InputValidation(class_name)
            check.dispatch(check.string_type, method.__name__, plot_name, 'plot_name')
            check.dispatch(check.index_range, method.__name__, feature, len(self.features))
            method(self, path, plot_name, feature=feature)
        return checker
    return wrap


# FeaturesConvexHullQuotient
def DisplayInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, plot_name, feature='all'):
            check = util.InputValidation(class_name)
            check.dispatch(check.string_type, method.__name__, plot_name, 'plot_name')
            check.dispatch(check.index_range, method.__name__, feature, len(self.features))
            method(self, plot_name, feature=feature)
        return checker
    return wrap
