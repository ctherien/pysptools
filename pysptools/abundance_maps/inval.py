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


def MapInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, U, normalize=False, mask=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.lib_type, method.__name__, U, 'U')
            check.dispatch(check.bool_type, method.__name__, normalize, 'normalize')
            check.dispatch(check.mask_type, method.__name__, mask)
            return method(self, M, U, normalize=normalize, mask=mask)
        return checker
    return wrap

def PlotInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, path, mask=None, interpolation='none', colorMap='jet', columns=None, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.mask_type, method.__name__, mask)
            check.dispatch(check.columns_type, method.__name__, columns, 'columns')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, path, mask=mask, interpolation=interpolation, colorMap=colorMap, columns=columns, suffix=suffix)
        return checker
    return wrap


def DisplayInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, mask=None, interpolation='none', colorMap='jet', columns=None, suffix=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.mask_type, method.__name__, mask)
            check.dispatch(check.columns_type, method.__name__, columns, 'columns')
            check.dispatch(check.suffix_type, method.__name__, suffix)
            method(self, mask=mask, interpolation=interpolation, colorMap=colorMap, columns=columns, suffix=suffix)
        return checker
    return wrap
