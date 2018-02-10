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
# all_tests.py - This file is part of the PySptools package.
#

"""
Call all the test_*.py
"""

from __future__ import print_function

import sys
from pysptools.tests import test_cls, test_detect, test_dnoise, test_vd, test_hull, \
     test_pwc, test_spectro, test_eea, test_HyperSVC, test_skl, \
     test_skl_multi_labels


def all_tests():
    try:
        print('====> testing test_cls')
        test_cls.tests()
    except Exception:
        print('************ Error in test: test_cls')
    try:
        print('====> testing test_detect')
        test_detect.tests()
    except Exception:
        print('************ Error in test: test_detect')
    try:
        print('====> testing test_dnoise')
        test_dnoise.tests()
    except Exception:
        print('************ Error in test: test_dnoise')
    try:
        print('====> testing test_vd')
        test_vd.tests()
    except Exception:
        print('************ Error in test: test_vd')
    try:
        print('====> testing test_hull')
        test_hull.tests()
    except Exception:
        print('************ Error in test: test_hull')
    try:
        print('====> testing test_kmeans')
        test_pwc.tests()
    except Exception:
        print('************ Error in test: test_kmeans')
    try:
        print('====> testing test_pwc')
        test_pwc.tests()
    except Exception:
        print('************ Error in test: test_pwc')
    try:
        print('====> testing test_spectro')
        test_spectro.tests()
    except Exception:
        print('************ Error in test: test_spectro')
    try:
        print('====> testing test_eea')
        test_eea.tests()
    except Exception:
        print('************ Error in test: test_eea')
    try:
        print('====> testing test_HyperSVC')
        test_HyperSVC.tests()
    except Exception:
        print('************ Error in test: test_HyperSVC')
    try:
        print('====> testing test_skl')
        test_skl.tests()
    except Exception:
        print('************ Error in test: test_skl')
    try:
        print('====> testing test_skl_multi_labels')
        test_skl_multi_labels.tests()
    except Exception:
        print('************ Error in test: test_skl_multi_labels')


if __name__ == '__main__':
    import sys
    all_tests()
