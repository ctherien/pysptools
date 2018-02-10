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
# input_vld.py - This file is part of the PySptools package.
#

import numpy as np


def simple_decorator(decorator):
    """ A well behaved decorator.
        Ref. wiki.python.org
    """
    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g
    # Now a few lines needed to make simple_decorator itself
    # be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator


class InputValidation(object):
    """ Validate method inputs. """

    # input_type

    axes_type       = 1
    bool_type       = 2
    cmap_exist      = 3
    columns_type    = 4
    cube_type       = 5
    cube_type2      = 6 # more restricted, for HySime
    far_type        = 7 # for HfcVd
    float_type      = 8
    float_type2     = 9
    index_range     = 10
    int_type        = 11
    int_type2       = 12 # int or None
    labels_type     = 13
    lib_type        = 14
    list_type       = 15
    mask_type       = 16
    spectrum_length = 17
    spectrum_type   = 18
    string_type     = 19
    string_type2    = 20
    suffix_type     = 21
    threshold_type  = 22
    threshold_type2 = 23
    threshold_type3 = 24 # for AbundanceClassification
    transform_type  = 25

    def __init__(self, class_id):
        self.class_id = class_id

    def dispatch(self, check_type, *args):
        if check_type == self.cube_type: self._cube_type(args)
        if check_type == self.lib_type: self._lib_type(args)
        if check_type == self.threshold_type: self._threshold_type(args)
        if check_type == self.cmap_exist: self._cmap_exist(args)
        if check_type == self.suffix_type: self._suffix_type(args)
        if check_type == self.index_range: self._index_range(args)
        if check_type == self.bool_type: self._bool_type(args)
        if check_type == self.spectrum_length: self._spectrum_length(args)
        if check_type == self.labels_type: self._labels_type(args)
        if check_type == self.mask_type: self._mask_type(args)
        if check_type == self.spectrum_type: self._spectrum_type(args)
        if check_type == self.threshold_type2: self._threshold_type2(args)
        if check_type == self.int_type: self._int_type(args)
        if check_type == self.axes_type: self._axes_type(args)
        if check_type == self.transform_type: self._transform_type(args)
        if check_type == self.int_type2: self._int_type2(args)
        if check_type == self.cube_type2: self._cube_type2(args)
        if check_type == self.far_type: self._far_type(args)
        if check_type == self.threshold_type3: self._threshold_type3(args)
        if check_type == self.string_type: self._string_type(args)
        if check_type == self.string_type2: self._string_type2(args)
        if check_type == self.list_type: self._list_type(args)
        if check_type == self.float_type: self._float_type(args)
        if check_type == self.float_type2: self._float_type2(args)
        if check_type == self.columns_type: self._int_type2(args)

    def _cube_type(self, args):
        method = args[0]
        M = args[1]
        obj_id = args[2]
        if type(M) is not np.ndarray:
            err1 = 'in {0}.{1}(), {2} is not a numpy.array'
            raise TypeError(err1.format(self.class_id, method, obj_id))
        if M.ndim != 3:
            err2 = 'in {0}.{1}(), {2} have {3} dimension(s), expected 3 dimensions'
            raise RuntimeError(err2.format(self.class_id, method, obj_id, M.ndim))

    def _lib_type(self, args):
        method = args[0]
        E = args[1]
        obj_id = args[2]
        if type(E) is not np.ndarray:
            err1 = 'in {0}.{1}(), {2} is not a numpy.array'
            raise TypeError(err1.format(self.class_id, method, obj_id))
        if E.ndim != 1 and E.ndim != 2:
            err2 = 'in {0}.{1}(), {2} have {3} dimension(s), expected 2 dimensions'
            raise RuntimeError(err2.format(self.class_id, method, obj_id, E.ndim))

    def _threshold_type(self, args):
        method = args[0]
        E = args[1]
        threshold = args[2]
        if type(threshold) is not float and type(threshold) is not list:
            err1 = 'in {0}.{1}(), threshold have {2}, expected float or list type'
            raise TypeError(err1.format(self.class_id, method, type(threshold)))
        if type(threshold) is list and len(threshold) != E.shape[0]:
            err2 = 'in {0}.{1}(), threshold have length {2}, expected length {3}'
            raise ValueError(err2.format(self.class_id, method, len(threshold), E.shape[0]))
        if type(threshold) is float:
            err3 = 'in {0}.{1}(), threshold value is {2}, expected value between 0.0 and 1.0'
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError(err3.format(self.class_id, method, threshold))
        if type(threshold) is list:
            err4 = 'in {0}.{1}(), threshold value is {2} at index {3}, expected value between 0.0 and 1.0'
            for i in range(len(threshold)):
                if threshold[i] < 0.0 or threshold[i] > 1.0:
                    raise ValueError(err4.format(self.class_id, method, threshold[i], i))

    def _spectrum_length(self, args):
        method = args[0]
        M = args[1]
        M_id = args[2]
        E = args[3]
        E_id = args[4]
        if E.ndim == 1:
            if M.shape[2] != E.shape[0]:
                err1 = 'in {0}.{1}(), the {2} spectrum length is different to the {3} spectrum length'
                raise RuntimeError(err1.format(self.class_id, method, M_id, E_id))
        if E.ndim == 2:
            if M.shape[2] != E.shape[1]:
                err2 = 'in {0}.{1}(), the {2} spectrum length is different to the {3} spectrum length'
                raise RuntimeError(err2.format(self.class_id, method, M_id, E_id))

    def _cmap_exist(self, args):
        method = args[0]
        cmap = args[1]
        to_call = args[2]
        # = args[2]
        if not isinstance(cmap, np.ndarray):
            err1 = 'in {0}.{1}(), call {2} before calling {3}'
            raise RuntimeError(err1.format(self.class_id, method, to_call, method))

    def _suffix_type(self, args):
        method = args[0]
        suffix = args[1]
        if type(suffix) is not str and suffix != None:
            err1 = 'in {0}.{1}(), suffix is not of str type or None'
            raise TypeError(err1.format(self.class_id, method))

    def _bool_type(self, args):
        method = args[0]
        obj = args[1]
        obj_id = args[2]
        if type(obj) is not bool:
            err = 'in {0}.{1}(), {2} is not of bool type'
            raise TypeError(err.format(self.class_id, method, obj_id))

    def _index_range(self, args):
        method = args[0]
        idx = args[1]
        n_em = args[2]
        if idx == 'all': return
        if idx < 1 or idx > n_em:
            err1 = 'in {0}.{1}(), indexing at {2} is out of range, expected a value between 1 and {3} or the string \'all\''
            raise IndexError(err1.format(self.class_id, method, idx, n_em))

    def _labels_type(self, args):
        method = args[0]
        labels = args[1]
        if type(labels) is not list and labels != None:
            err1 = 'in {0}.{1}(), labels is not a list or None'
            raise TypeError(err1.format(self.class_id, method))

    def _mask_type(self, args):
        method = args[0]
        mask = args[1]
        if type(mask) is not np.ndarray and mask != None:
            err1 = 'in {0}.{1}(), mask is not a numpy.array'
            raise TypeError(err1.format(self.class_id, method))
        if isinstance(mask, np.ndarray) and mask.ndim != 2:
            err2 = 'in {0}.{1}(), mask have {2} dimension(s), expected 2 dimensions'
            raise RuntimeError(err2.format(self.class_id, method, mask.ndim))

    def _spectrum_type(self, args):
        method = args[0]
        t = args[1]
        obj_id = args[2]
        if type(t) is not np.ndarray:
            err1 = 'in {0}.{1}(), {2} is not a numpy.array'
            raise TypeError(err1.format(self.class_id, method, obj_id))
        if t.ndim != 1:
            err2 = 'in {0}.{1}(), {2} have {3} dimension(s), a vector is expected'
            raise RuntimeError(err2.format(self.class_id, method, obj_id, t.ndim))

    def _threshold_type2(self, args):
        method = args[0]
        threshold = args[1]
        if type(threshold) is not float and threshold != None:
            err1 = 'in {0}.{1}(), threshold is not of float type or None'
            raise TypeError(err1.format(self.class_id, method))

    def _int_type(self, args):
        method = args[0]
        number = args[1]
        obj_id = args[2]
        if type(number) is not int:
            err1 = 'in {0}.{1}(), {2} is not of int type'
            raise TypeError(err1.format(self.class_id, method, obj_id))

    def _axes_type(self, args):
        method = args[0]
        axes = args[1]
        if type(axes) is not dict and axes != None:
            err1 = 'in {0}.{1}(), axes is not of dict type or None'
            raise TypeError(err1.format(self.class_id, method))

    def _transform_type(self, args):
        method = args[0]
        q = args[1]
        transform = args[2]
        if transform == None: return
        if type(transform) is not np.ndarray:
            err1 = 'in {0}.{1}(), transform is not a numpy.array'
            raise RuntimeError(err1.format(self.class_id, method))
        if transform.ndim != 3:
            err2 = 'in {0}.{1}(), transform have {2} dimension(s), expected 3 dimensions'
            raise RuntimeError(err2.format(self.class_id, method, transform.ndim))
        if q-1 != transform.shape[2]:
            err3 = 'in {0}.{1}(), q equal {2} and transform have {3} components, expected q == components - 1'
            raise RuntimeError(err3.format(self.class_id, method, q, transform.shape[2]))

    def _int_type2(self, args):
        method = args[0]
        number = args[1]
        obj_id = args[2]
        if type(number) is not int and number != None:
            err1 = 'in {0}.{1}(), {2} is not of int type or None'
            raise TypeError(err1.format(self.class_id, method, obj_id))

    def _cube_type2(self, args):
        method = args[0]
        M = args[1]
        obj_id = args[2]
        if type(M) is not np.ndarray:
            err1 = 'in {0}.{1}(), {2} is not a numpy.array'
            raise TypeError(err1.format(self.class_id, method, obj_id))
        if M.ndim != 3:
            err2 = 'in {0}.{1}(), {2} have {3} dimension(s), expected 3 dimensions'
            raise RuntimeError(err2.format(self.class_id, method, obj_id, M.ndim))
        h, w, numBands = M.shape
        if numBands < 2:
            err3 = 'in {0}.{1}(), too few bands to estimate the noise'
            raise RuntimeError(err3.format(self.class_id, method))

    def _far_type(self, args):
        method = args[0]
        far = args[1]
        obj_id = args[2]
        if not (type(far) is list or far == 'default'):
            err1 = 'in {0}.{1}(), far have {2}, expected list type'
            raise TypeError(err1.format(self.class_id, method, type(far)))
        if type(far) is list:
            for i in range(len(far)):
                if far[i] > 1.0:
                    err2 = 'in {0}.{1}(), far value is {2} at index {3}, expected value < 1'
                    raise ValueError(err2.format(self.class_id, method, far[i], i))

    def _threshold_type3(self, args):
        method = args[0]
        M = args[1]
        threshold = args[2]
        if type(threshold) is not float and type(threshold) is not list:
            err1 = 'in {0}.{1}(), threshold have {2}, expected float or list type'
            raise TypeError(err1.format(self.class_id, method, type(threshold)))
        if type(threshold) is list and len(threshold) != M.shape[2]:
            err2 = 'in {0}.{1}(), threshold have length {2}, expected length {3}'
            raise ValueError(err2.format(self.class_id, method, len(threshold), M.shape[2]))
        if type(threshold) is float:
            err3 = 'in {0}.{1}(), threshold value is {2}, expected value between 0.0 and 1.0'
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError(err3.format(self.class_id, method, threshold))
        if type(threshold) is list:
            err4 = 'in {0}.{1}(), threshold value is {2} at index {3}, expected value between 0.0 and 1.0'
            for i in range(len(threshold)):
                if threshold[i] < 0.0 or threshold[i] > 1.0:
                    raise ValueError(err4.format(self.class_id, method, threshold[i], i))

    def _string_type(self, args):
        method = args[0]
        string = args[1]
        obj_id = args[2]
        if type(string) is not str:
            err1 = 'in {0}.{1}(), {2} is not of string type'
            raise TypeError(err1.format(self.class_id, method, obj_id))

    def _string_type2(self, args):
        method = args[0]
        string = args[1]
        obj_id = args[2]
        if type(string) is not str and string != None:
            err1 = 'in {0}.{1}(), {2} is not of string type or None'
            raise TypeError(err1.format(self.class_id, method, obj_id))

    def _list_type(self, args):
        method = args[0]
        lst = args[1]
        obj_id = args[2]
        if type(lst) is not list:
            err1 = 'in {0}.{1}(), {2} is not of list type'
            raise TypeError(err1.format(self.class_id, method, obj_id))

    def _float_type(self, args):
        method = args[0]
        flt = args[1]
        obj_id = args[2]
        if type(flt) is not float:
            err1 = 'in {0}.{1}(), {2} is not of float type'
            raise TypeError(err1.format(self.class_id, method, obj_id))

    def _float_type2(self, args):
        method = args[0]
        flt = args[1]
        obj_id = args[2]
        if type(flt) is not float and flt != None:
            err1 = 'in {0}.{1}(), {2} is not of float type or None'
            raise TypeError(err1.format(self.class_id, method, obj_id))
