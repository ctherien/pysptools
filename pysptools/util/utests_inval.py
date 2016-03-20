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

import unittest
import numpy as np
from pysptools.util.input_vld import InputValidation
from types import *


class TestInputValidation(unittest.TestCase):

    def setUp(self):
        self.er = InputValidation('CLASS_ID')

    def test_cube_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.cube_type, 'method_id', [0,1], 'M')
        # err2
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.cube_type, 'method_id', np.zeros((2,2)), 'M')

    def test_lib_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.lib_type, 'method_id', [0,1], 'E')
        # err2
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.lib_type, 'method_id', np.zeros((2,2,2)), 'E')

    def test_threshold_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.threshold_type, 'method_id', np.zeros((2,2)), 'string')
        # err2
        with self.assertRaises(ValueError):
            self.er.dispatch(self.er.threshold_type, 'method_id', np.zeros((2,2)), [1,2,3])
        # err3
        with self.assertRaises(ValueError):
            self.er.dispatch(self.er.threshold_type, 'method_id', np.zeros((2,2)), 2.5)
        # err4
        with self.assertRaises(ValueError):
            self.er.dispatch(self.er.threshold_type, 'method_id', np.zeros((4,2)), [0.1,0.1,2.4,0.1])

    def test_spectrum_length(self):
        # err1
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.spectrum_length, 'method_id', np.zeros((2,2,2)), 'N', np.zeros((3)), 'E')
        # err2
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.spectrum_length, 'method_id', np.zeros((2,2,2)), 'N', np.zeros((3,3)), 'E')

    def test_suffix_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.suffix_type, 'method_id', 1.0)

    def test_bool_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.bool_type, 'method_id', 1.0, 'BOOL_OBJ')

    def test_labels_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.labels_type, 'method_id', 'bad')

    def test_mask_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.mask_type, 'method_id', 1.0)
        # err2
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.mask_type, 'method_id', np.zeros((2,2,2)))

    def test_index_range(self):
        # err1
        with self.assertRaises(IndexError):
            self.er.dispatch(self.er.index_range, 'method_id', 4, 3)

    def test_cmap_exist(self):
        # err1
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.cmap_exist, 'method_id', None, 'to_call')

    def test_spectrum_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.spectrum_type, 'method_id', [0,1], 't')
        # err2
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.spectrum_type, 'method_id', np.zeros((2,2,2)), 't')

    def test_threshold_type2(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.threshold_type2, 'method_id', 'string')

    def test_int_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.int_type, 'method_id', 'string', 'INT_OBJ')

    def test_axes_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.axes_type, 'method_id', 'string')

    def test_transform_type(self):
        # err1
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.transform_type, 'method_id', 2, [1,2])
        # err2
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.transform_type, 'method_id', 2, np.zeros((2,2)))
        # err3
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.transform_type, 'method_id', 2, np.zeros((2,2,2)))

    def test_int_type2(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.int_type2, 'method_id', 'string', 'INT_OBJ')

    def test_cube_type2(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.cube_type2, 'method_id', [0,1], 'M')
        # err2
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.cube_type2, 'method_id', np.zeros((2,2)), 'M')
        # err3
        with self.assertRaises(RuntimeError):
            self.er.dispatch(self.er.cube_type2, 'method_id', np.zeros((1,1,1)), 'M')

    def test_far_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.far_type, 'method_id', 2, 'far')
        # err2
        with self.assertRaises(ValueError):
            self.er.dispatch(self.er.far_type, 'method_id', [0.1,2,0.2], 'far')

    def test_threshold_type3(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.threshold_type3, 'method_id', np.zeros((2,2)), 'string')
        # err2
        with self.assertRaises(ValueError):
            self.er.dispatch(self.er.threshold_type3, 'method_id', np.zeros((2,2,4)), [1,2,3])
        # err3
        with self.assertRaises(ValueError):
            self.er.dispatch(self.er.threshold_type3, 'method_id', np.zeros((2,2,1)), 2.5)
        # err4
        with self.assertRaises(ValueError):
            self.er.dispatch(self.er.threshold_type3, 'method_id', np.zeros((2,2,4)), [0.1,0.1,2.4,0.1])

    def test_string_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.string_type, 'method_id', 2, 'STRING_OBJ')

    def test_string_type2(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.string_type2, 'method_id', 2, 'STRING_OR_NONE_OBJ')

    def test_list_type(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.list_type, 'method_id', 2, 'LIST_OBJ')

    def test_float_type(self):
        # err1
        #with self.assertRaises(TypeError):
            self.er.dispatch(self.er.float_type, 'method_id', 2, 'FLOAT_OBJ')

    def test_float_type2(self):
        # err1
        with self.assertRaises(TypeError):
            self.er.dispatch(self.er.float_type2, 'method_id', 2, 'FLOAT_OR_NONE_OBJ')

if __name__ == '__main__':
    unittest.main()
