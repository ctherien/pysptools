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

# SavitzkyGolay
def DenoiseSpectraInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, window_size, order, deriv=0, rate=1):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.int_type, method.__name__, window_size, 'window_size')
            check.dispatch(check.int_type, method.__name__, order, 'order')
            check.dispatch(check.int_type, method.__name__, deriv, 'deriv')
            check.dispatch(check.int_type, method.__name__, rate, 'rate')
            return method(self,  M, window_size, order, deriv=deriv, rate=rate)
        return checker
    return wrap


# SavitzkyGolay
def DenoiseBandsInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, window_size, order, derivative=None):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.int_type, method.__name__, window_size, 'window_size')
            check.dispatch(check.int_type, method.__name__, order, 'order')
            check.dispatch(check.string_type2, method.__name__, derivative, 'derivative')
            return method(self, M, window_size, order, derivative=derivative)
        return checker
    return wrap


# Whiten
def ApplyInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            return method(self, M)
        return checker
    return wrap


# Whiten
def XInputValidation(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, X):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, X, 'X')
            return method(self, X)
        return checker
    return wrap

