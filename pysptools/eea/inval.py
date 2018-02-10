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

# PPI
def ExtractInputValidation1(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, q, numSkewers=10000, normalize=False, mask=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.int_type, method.__name__, q, 'q')
            check.dispatch(check.int_type, method.__name__, numSkewers, 'numSkewers')
            check.dispatch(check.bool_type, method.__name__, normalize, 'normalize')
            check.dispatch(check.mask_type, method.__name__, mask)
            return method(self, M, q, numSkewers=numSkewers, normalize=normalize, mask=mask)
        return checker
    return wrap


# NFINDR
def ExtractInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, q, transform=None, maxit=None, normalize=False, ATGP_init=False, mask=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.int_type, method.__name__, q, 'q')
            check.dispatch(check.transform_type, method.__name__, q, transform)
            check.dispatch(check.int_type2, method.__name__, maxit, 'maxit')
            check.dispatch(check.bool_type, method.__name__, normalize, 'normalize')
            check.dispatch(check.bool_type, method.__name__, ATGP_init, 'ATGP_init')
            check.dispatch(check.mask_type, method.__name__, mask)
            return method(self, M, q, transform=transform, maxit=maxit, normalize=normalize, ATGP_init=ATGP_init, mask=mask)
        return checker
    return wrap


# ATGP
def ExtractInputValidation3(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, q, normalize=False, mask=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.int_type, method.__name__, q, 'q')
            check.dispatch(check.bool_type, method.__name__, normalize, 'normalize')
            check.dispatch(check.mask_type, method.__name__, mask)
            return method(self, M, q, normalize=normalize, mask=mask)
        return checker
    return wrap


# FIPPI
def ExtractInputValidation4(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, q=None, maxit=None, normalize=False, mask=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.int_type, method.__name__, q, 'q')
            check.dispatch(check.int_type2, method.__name__, maxit, 'maxit')
            check.dispatch(check.bool_type, method.__name__, normalize, 'normalize')
            check.dispatch(check.mask_type, method.__name__, mask)
            return method(self, M, q=q, maxit=maxit, normalize=normalize, mask=mask)
        return checker
    return wrap


def PlotInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, axes=None, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.axes_type, method.__name__, axes)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, axes=axes, suffix=suffix)
        return checker
    return wrap


def DisplayInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, axes=None, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.axes_type, method.__name__, axes)
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, axes=axes, suffix=suffix)
        return checker
    return wrap
