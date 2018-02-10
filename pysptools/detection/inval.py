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


# MatchedFilter, ACE, CEM, GLRT
def DetectInputValidation1(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, t, threshold=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.spectrum_type, method.__name__, t, 't')
            check.dispatch(check.threshold_type2, method.__name__, threshold)
            return method(self, M, t, threshold=threshold)
        return checker
    return wrap


# OSP
def DetectInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, E, t, threshold=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.lib_type, method.__name__, E, 'E')
            check.dispatch(check.spectrum_type, method.__name__, t, 't')
            check.dispatch(check.threshold_type2, method.__name__, threshold)
            return method(self, M, E, t, threshold=threshold)
        return checker
    return wrap


def PlotInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, whiteOnBlack=True, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.bool_type, method.__name__, whiteOnBlack, 'whiteOnBlack')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, whiteOnBlack=whiteOnBlack, suffix=suffix)
        return checker
    return wrap


def DisplayInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, whiteOnBlack=True, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.bool_type, method.__name__, whiteOnBlack, 'whiteOnBlack')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, whiteOnBlack=whiteOnBlack, suffix=suffix)
        return checker
    return wrap
