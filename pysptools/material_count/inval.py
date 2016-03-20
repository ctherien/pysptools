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


# HySime
def CountInputValidation1(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type2, method.__name__, M, 'M')
            return method(self, M)
        return checker
    return wrap


# HfcVd
def CountInputValidation2(class_name):
    @util.simple_decorator
    def wrap(method):
        def checker(self, M, far='default', noise_whitening=False):
            check = util.InputValidation(class_name)
            check.dispatch(check.cube_type, method.__name__, M, 'M')
            check.dispatch(check.far_type, method.__name__, far, 'far')
            check.dispatch(check.bool_type, method.__name__, noise_whitening, 'noise_whitening')
            return method(self, M, far=far, noise_whitening=noise_whitening)
        return checker
    return wrap
