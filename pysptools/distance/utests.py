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
# utests.py - This file is part of the PySptools package.
#

from __future__ import print_function

import unittest
import numpy as np
from pysptools.distance import SAM, SID, chebyshev, NormXCorr
from types import *

# todo
# test same length
# test numpy array ?, if not translate ?


class TestSAMFunction(unittest.TestCase):

    def setUp(self):
        self.vec1 = np.array([1,2,3,4,5])
        self.vec2 = np.array([6,7,8,9,10])
        self.dist = 0.26554161733900966
        #self.vec3 = np.array([1,2,3,4,5,9])

    def runTest(self):
        print('==> runTest: TestSAMFunction')
        self.test_distance()

    def test_distance(self):
        self.assertEqual(SAM(self.vec1,self.vec2), self.dist)


class TestSIDFunction(unittest.TestCase):

    def setUp(self):
        self.vec1 = np.array([1,2,3,4,5])
        self.vec2 = np.array([6,7,8,9,10])
        self.dist = 0.1099607220673022
        #self.vec3 = np.array([1,2,3,4,5,9])

    def runTest(self):
        print('==> runTest: TestSIDFunction')
        self.test_distance()

    def test_distance(self):
        self.assertEqual(SID(self.vec1,self.vec2), self.dist)


class TestChebyshevFunction(unittest.TestCase):

    def setUp(self):
        self.vec1 = np.array([1,2,3,4,5])
        self.vec2 = np.array([6,7,8,9,10])
        self.dist = 5
        #self.vec3 = np.array([1,2,3,4,5,9])

    def runTest(self):
        print('==> runTest: TestChebyshevFunction')
        self.test_distance()

    def test_distance(self):
        self.assertEqual(chebyshev(self.vec1,self.vec2), self.dist)


class TestNormXCorrFunction(unittest.TestCase):

    precision = 1000000

    def norm(self, f):
        return int(f * self.precision)

    def setUp(self):
        self.vec1 = np.array([1,2,3,4,5])
        self.vec2 = np.array([6,7,8,9,10])
        #self.vec2 = np.array([6,17,8,99,10])
        # compare with 0.999999
        self.dist = self.norm(0.999999)

    def runTest(self):
        print('==> runTest: TestNormXCorrFunction')
        self.test_distance()

    def test_distance(self):
        self.assertEqual(self.norm(NormXCorr(self.vec1,self.vec2)), self.dist)


##def safe_distance(fn, s1, s2):
##    """ Validate the inputs and call distance fn """
##    from types import *
##    if not(fn in (SAM, SID, chebyshev, NormXCorr)):
##        print 'BadFnName'
##        raise Exception
##    if type(s1) is ListType:
##        s1 = np.array(s1)
##    if type(s2) is ListType:
##        s2 = np.array(s2)
##    if type(s1) is np.ndarray and type(s2) is np.ndarray:
##        if (len(s1.shape) == 1 and len(s2.shape) == 1
##            and s1.shape[0] == s2.shape[0]):
##            return fn(s1,s2)
##        else:
##            print 'WrongArrayDimension'
##            raise Exception
##    else:
##        return fn(s1,s2)
##    print 'BadType'
##    raise Exception

# I dont now if I will keep this function,
# it seems pretty useless
def safe_distance(fn, s1, s2):
    """ Validate the inputs and call distance fn """
    if not(fn in (SAM, SID, chebyshev, NormXCorr)):
        print('BadFnName')
        raise Exception
    if isinstance(s1, list):
        s1 = np.array(s1)
    if isinstance(s2, list):
        s2 = np.array(s2)
    if isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
        if (len(s1.shape) == 1 and len(s2.shape) == 1
            and s1.shape[0] == s2.shape[0]):
            return fn(s1,s2)
        else:
            print('WrongArrayDimension')
            raise Exception
    else:
        return fn(s1,s2)
    print('BadType')
    raise Exception


class TestSafeDistance(unittest.TestCase):

    def setUp(self):
        self.vec1 = np.array([1,2,3,4,5])
        self.vec2 = [1,2,3,4,5]
        self.vec3 = [6,7,8,9,10]
        self.vec4 = np.array([6,7,8,9,10])
        self.vec5 = np.array([[6,7,8,9,10],[9,8,7,6,5]])
        self.vec6 = [10,11,12,13,14,15,17]
        self.dist = 0.26554161733900966

    def runTest(self):
        print('==> runTest: TestSafeDistance')
        self.test_BadFnName()
        self.test_WrongArrayDimension()
        self.test_BadType()
        self.test_distance()

    def test_BadFnName(self):
        self.assertRaises(Exception, safe_distance, (TestSAMFunction,self.vec1,self.vec2))

    def test_WrongArrayDimension(self):
        self.assertRaises(Exception, safe_distance, (SAM,self.vec1,self.vec6))

    def test_BadType(self):
        self.assertRaises(Exception, safe_distance, (SID,self.vec1, 5))

    def test_distance(self):
        self.assertEqual(safe_distance(SAM,self.vec1,self.vec3), self.dist)


if __name__ == '__main__':
    unittest.main()
