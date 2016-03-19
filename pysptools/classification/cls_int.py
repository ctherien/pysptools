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
# cls_int.py - This file is part of the PySptools package.
#

"""
Spectral Angle Mapper class
Spectral Information Divergence class
Normalized cross correlation class
"""

import numpy as np
from . import cls
from . import out
from .inval import *
from .docstring import *


def _normalize(M):
    """
    Normalizes M to be in range [0, 1].

    Parameters:
      M: `numpy array`
          1D, 2D or 3D data.

    Returns: `numpy array`
          Normalized data.
    """

    minVal = np.min(M)
    maxVal = np.max(M)

    Mn = M - minVal;

    if maxVal == minVal:
        return np.zeros(M.shape);
    else:
        return Mn / (maxVal-minVal)


def _single_value_min(data, threshold):
    """
    Use a threshold to extract the minimum value along
    the data y axis.
    """
    amin = np.min(data)
    amax = np.max(data)
    limit = amin + (amax - amin) * threshold
    return data < limit


def _single_value_max(data, threshold):
    """
    Use a threshold to extract the maximum value along
    the data y axis.
    """
    amin = np.min(data)
    amax = np.max(data)
    limit = amax - (amax - amin) * threshold
    return data > limit


def _simple_decorator(decorator):
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

@_simple_decorator
def Plot(method):
    def plot(self, path, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        if self.n_classes == 1:
            self.output.plot1(self.cmap, path=path, mask=mask, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        else:
            self.output.plot(self.cmap, self.n_classes, path=path, labels=labels, mask=mask, interpolation=interpolation, colorMap=colorMap, firstBlack=True, suffix=suffix)
    return plot


@_simple_decorator
def Display(method):
    def display(self, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        if self.n_classes == 1:
            self.output.plot1(self.cmap, mask=mask, interpolation=interpolation, colorMap=colorMap, suffix=suffix)
        else:
            self.output.plot(self.cmap, self.n_classes, labels=labels, mask=mask, interpolation=interpolation, colorMap=colorMap, firstBlack=True, suffix=suffix)
    return display


def _document(cls):
    import sys
    if sys.version_info[0] == 2:
        cls.classify.__func__.__doc__ = classify_docstring
        cls.get_single_map.__func__.__doc__ = get_single_map_docstring
        cls.plot_single_map.__func__.__doc__ = plot_single_map_docstring
        cls.display_single_map.__func__.__doc__ = display_single_map_docstring
        cls.plot.__func__.__doc__ = plot_docstring
        cls.display.__func__.__doc__ = display_docstring
        cls.plot_histo.__func__.__doc__ = plot_histo_docstring
    if sys.version_info[0] == 3:
        cls.classify.__doc__ = classify_docstring
        cls.get_single_map.__doc__ = get_single_map_docstring
        cls.plot_single_map.__doc__ = plot_single_map_docstring
        cls.display_single_map.__doc__ = display_single_map_docstring
        cls.plot.__doc__ = plot_docstring
        cls.display.__doc__ = display_docstring
        cls.plot_histo.__doc__ = plot_histo_docstring


def _compress(vec, mask):
    n = np.sum(mask)
    cmp = np.ndarray((n, vec.shape[1]), dtype=np.float)
    i = 0
    for j in range(mask.shape[0]):
        if mask[j] == 1:
            cmp[i] = vec[j]
            i += 1
    return cmp


def _expand(amap, mask, l):
    exp = np.zeros(l, dtype=np.float)
    i = 0
    for j in range(mask.shape[0]):
        if mask[j] == 1:
            exp[j] = amap[i]
            i += 1
    return exp


def _expand2(amap, mask, l, q):
    exp = np.zeros((l,q), dtype=np.float)
    i = 0
    for j in range(mask.shape[0]):
        if mask[j] == 1:
            exp[j] = amap[i]
            i += 1
    return exp


class SAM(object):
    """Classify a HSI cube using the spectral angle mapper algorithm
    and a spectral library."""

    def __init__(self):
        self.output = out.Output('SAM')
        self.cmap = None
        self.angles = None
        # spectra number
        self.n_classes = None
        # a float or a list of float
        self.threshold = None

    @ClassifyInputValidation('SAM')
    def classify(self, M, E, threshold=0.1, mask=None):
        if E.ndim == 1:
            self.n_classes = 1
        else:
            self.n_classes = E.shape[0]
        self.threshold = threshold
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        M = np.reshape(M, (w*h, numBands))
        Mn = _normalize(M)
        En = _normalize(E)

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, w*h)
            cMn = _compress(Mn, m)
        else:
            cMn = Mn

        if E.ndim == 1:
            cmap, angles = self._class_single_pixel(cMn, En, threshold)
        else:
            cmap, angles = cls.SAM_classifier(cMn, En, threshold)

        if isinstance(mask, np.ndarray):
            cmap = _expand(cmap, m, w*h)
            if  self.n_classes > 1:
                angles = _expand2(angles, m, w*h, E.shape[0])
            elif self.n_classes == 1:
                angles = _expand(angles, m, w*h)

        self.cmap = np.reshape(cmap, (h, w))
        self.angles = np.reshape(angles, (h, w, self.n_classes))
        return self.cmap

    def __str__(self):
        return 'pysptools.classification.cls_int.SAM object, hcube: {0}x{1}x{2}, n classes: {3}'.format(self.h, self.w, self.numBands, self.n_classes)

    def _class_single_pixel(self, M, E, threshold):
        import pysptools.distance as dst
        angles = np.ndarray((M.shape[0], 1), dtype=np.float)
        for i in range(M.shape[0]):
            angles[i] = dst.SAM(M[i], E)
        cmap = _single_value_min(angles, threshold) * angles
        return cmap, angles

    @GetMapInputValidation('SAM', 'classify')
    def get_angles_map(self):
        """
        Returns: `numpy array`
            The angles array (m x n x spectra number).
        """
        return self.angles

    @GetMapInputValidation('SAM', 'classify')
    def get_angles_stats(self):
        """
        Returns: `dic`
             Angles stats.
        """
        mm = {}
        for i in range(self.n_classes):
            mm[i] = (np.min(self.angles[:,:,i]),
                     np.max(self.angles[:,:,i]))
        return mm

    @GetSingleMapInputValidation('SAM')
    def get_single_map(self, lib_idx, constrained=True):
        if self.n_classes == 1:
            return None
        return self.output.get_single_map(lib_idx, self.cmap, self.angles, self.threshold, constrained, stretch=False, inverse_scale=False)

    @PlotSingleMapInputValidation('SAM', 'classify')
    def plot_single_map(self, path, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
        self.output.plot_single_map(path, self.cmap, self.angles, lib_idx, self.n_classes, self.threshold, constrained, stretch, colorMap, suffix)

    @DisplaySingleMapInputValidation('SAM', 'classify')
    def display_single_map(self, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
        self.output.plot_single_map(None, self.cmap, self.angles, lib_idx, self.n_classes, self.threshold, constrained, stretch, colorMap, suffix)

    @PlotInputValidation('SAM')
    @Plot
    def plot(self, path, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

    @DisplayInputValidation('SAM')
    @Display
    def display(self, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

    @PlotHistoInputValidation('SAM')
    def plot_histo(self, path, suffix=None):
        self.output.plot_histo(path, self.cmap, self.n_classes, suffix)

_document(SAM)


class SID(object):
    """Classify a HSI cube using the spectral information divergence
    algorithm and a spectral library."""

    def __init__(self):
        self.output = out.Output('SID')
        self.cmap = None
        self.sid = None
        # endmembers number
        self.n_classes = None
        self.threshold = None

    @ClassifyInputValidation('SID')
    def classify(self, M, E, threshold=0.1, mask=None):
        if E.ndim == 1:
            self.n_classes = 1
        else:
            self.n_classes = E.shape[0]
        self.threshold = threshold
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        M = np.reshape(M, (w*h, numBands))
        Mn = _normalize(M)
        En = _normalize(E)

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, w*h)
            cMn = _compress(Mn, m)
        else:
            cMn = Mn

        if E.ndim == 1:
            cmap, sid = self._class_single_pixel(cMn, En, threshold)
        else:
            cmap, sid = cls.SID_classifier(cMn, En, threshold)

        if isinstance(mask, np.ndarray):
            cmap = _expand(cmap, m, w*h)
            if  self.n_classes > 1:
                sid = _expand2(sid, m, w*h, E.shape[0])
            elif self.n_classes == 1:
                sid = _expand(sid, m, w*h)

        self.cmap = np.reshape(cmap, (h, w))
        self.sid = np.reshape(sid, (h, w, self.n_classes))
        return self.cmap

    def __str__(self):
        return 'pysptools.classification.cls_int.SID object, hcube: {0}x{1}x{2}, n classes: {3}'.format(self.h, self.w, self.numBands, self.n_classes)

    def _class_single_pixel(self, M, E, threshold):
        import pysptools.distance as dst
        sid = np.ndarray((M.shape[0], 1), dtype=np.float)
        for i in range(M.shape[0]):
            sid[i] = dst.SID(M[i], E)
        cmap = _single_value_min(sid, threshold) * sid
        return cmap, sid

    @GetMapInputValidation('SID', 'classify')
    def get_SID_map(self):
        """
        Returns: `numpy array`
            The SID array (m x n x spectra number).
        """
        return self.sid

    @GetSingleMapInputValidation('SID')
    def get_single_map(self, lib_idx, constrained=True):
        if self.n_classes == 1:
            return None
        return self.output.get_single_map(lib_idx, self.cmap, self.sid, self.threshold, constrained, stretch=False, inverse_scale=False)

    @PlotSingleMapInputValidation('SID', 'classify')
    def plot_single_map(self, path, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
        self.output.plot_single_map(path, self.cmap, self.sid, lib_idx, self.n_classes, self.threshold, constrained, stretch, colorMap, suffix)

    @DisplaySingleMapInputValidation('SID', 'classify')
    def display_single_map(self, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
        self.output.plot_single_map(None, self.cmap, self.sid, lib_idx, self.n_classes, self.threshold, constrained, stretch, colorMap, suffix)

    @PlotInputValidation('SID')
    @Plot
    def plot(self, path, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

    @DisplayInputValidation('SID')
    @Display
    def display(self, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

    @PlotHistoInputValidation('SID')
    def plot_histo(self, path, suffix=None):
        self.output.plot_histo(path, self.cmap, self.n_classes, suffix)

_document(SID)


class NormXCorr(object):
    """Classify a HSI cube using the normalized cross correlation
    algorithm and a spectral library."""

    def __init__(self):
        self.output = out.Output('NormXCorr')
        self.cmap = None
        self.corr = None
        # endmembers number
        self.n_classes = None
        self.threshold = None

    def __str__(self):
        return 'pysptools.classification.cls_int.NormXCorr object, hcube: {0}x{1}x{2}, n classes: {3}'.format(self.h, self.w, self.numBands, self.n_classes)

    @ClassifyInputValidation('NormXCorr')
    def classify(self, M, E, threshold=0.1, mask=None):
        if E.ndim == 1:
            self.n_classes = 1
        else:
            self.n_classes = E.shape[0]
        if type(threshold) == float:
            self.threshold = 1 - threshold
        if type(threshold) == list:
            self.threshold = [1 - x for x in threshold]
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        M = np.reshape(M, (w*h, numBands))
        Mn = _normalize(M)
        En = _normalize(E)

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, w*h)
            cMn = _compress(Mn, m)
        else:
            cMn = Mn

        if E.ndim == 1:
            cmap, corr = self._class_single_pixel(cMn, En, threshold)
        else:
            cmap, corr = cls.NormXCorr_classifier(cMn, En, threshold)

        if isinstance(mask, np.ndarray):
            cmap = _expand(cmap, m, w*h)
            if  self.n_classes > 1:
                corr = _expand2(corr, m, w*h, E.shape[0])
            elif self.n_classes == 1:
                corr = _expand(corr, m, w*h)

        self.cmap = np.reshape(cmap, (h, w))
        self.corr = np.reshape(corr, (h, w, self.n_classes))
        return self.cmap

    def _class_single_pixel(self, M, E, threshold):
        import pysptools.distance as dst
        corr = np.ndarray((M.shape[0], 1), dtype=np.float)
        for i in range(M.shape[0]):
            corr[i] = dst.NormXCorr(M[i], E)
        cmap = _single_value_max(corr, threshold) * corr
        return cmap, corr

    @GetMapInputValidation('NormXCorr', 'classify')
    def get_NormXCorr_map(self):
        """
        Returns: `numpy array`
            The NormXCorr array (m x n x spectra number).
        """
        return self.corr

    @GetSingleMapInputValidation('NormXCorr')
    def get_single_map(self, lib_idx, constrained=True):
        if self.n_classes == 1:
            return None
        return self.output.get_single_map(lib_idx, self.cmap, self.corr, self.threshold, constrained, stretch=False, inverse_scale=False)

    @PlotSingleMapInputValidation('NormXCorr', 'classify')
    def plot_single_map(self, path, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
        self.output.plot_single_map(path, self.cmap, self.corr, lib_idx, self.n_classes, self.threshold, constrained, stretch, colorMap, suffix)

    @DisplaySingleMapInputValidation('NormXCorr', 'classify')
    def display_single_map(self, lib_idx, constrained=True, stretch=False, colorMap='spectral', suffix=None):
        self.output.plot_single_map(None, self.cmap, self.corr, lib_idx, self.n_classes, self.threshold, constrained, stretch, colorMap, suffix)

    @PlotInputValidation('NormXCorr')
    @Plot
    def plot(self, path, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

    @DisplayInputValidation('NormXCorr')
    @Display
    def display(self, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

    @PlotHistoInputValidation('NormXCorr')
    def plot_histo(self, path, suffix=None):
        self.output.plot_histo(path, self.cmap, self.n_classes, suffix)

_document(NormXCorr)


class AbundanceClassification(object):
    """Classify abundance maps."""

    def __init__(self):
        self.output = out.Output('AbundanceClassification')
        self.cmap = None
        self.n_classes = None
        self.threshold = None

    @ClassifyInputValidation2('AbundanceClassification')
    def classify(self, amaps, threshold=0.1, mask=None):
        """
        Use a group of abundance maps generated by UCLS, NNLS or FCLS,
        to construct a classification map.

        Parameters:
            amaps: `numpy array`
              A HSI cube (m x n x p).

            threshold: `float [default 0.1] or list`
             * If float, threshold is applied on all the spectra.
             * If a list, individual threshold is applied on each
               spectrum, in this case the list must have the same
               number of threshold values than the number of spectra.
             * Threshold have values between 0.0 and 1.0.

        Returns: `numpy array`
              A class map (m x n x 1).
        """
        self.n_classes = amaps.shape[2]
        if type(threshold) == float:
            self.threshold = 1 - threshold
        if type(threshold) == list:
            self.threshold = [1 - x for x in threshold]
        h, w, n_maps = amaps.shape
        self.h, self.w, self.n_maps = amaps.shape
        amaps = np.reshape(amaps, (w*h, n_maps))
        amaps = _normalize(amaps)
        if self.n_classes == 1:
            cmap = self._class_single_pixel(amaps, self.threshold)
        else:
            cmap = self._dispatch(amaps, self.threshold)
        self.cmap = np.reshape(cmap, (h, w))
        return self.cmap

    def __str__(self):
        return 'pysptools.classification.cls_int.AbundanceClassification object, amaps: {0}x{1}x{2}, n classes: {3}'.format(self.h, self.w, self.n_maps, self.n_classes)

    def _dispatch(self, maps, threshold):
        if isinstance(threshold, float):
            cmap = self._single_value_max(maps, threshold)
        elif isinstance(threshold, list):
            cmap = self._multiple_values_max(maps, threshold)
        return cmap

    def _class_single_pixel(self, m, threshold):
        cmap = self._single_value_max(m, threshold)
        return cmap

    def _single_value_max(self, maps, threshold):
        """
        Use a threshold to extract the maximum value along
        the y axis.
        """
        max_vec = np.max(maps, axis=1)
        cmin = np.min(max_vec)
        cmax = np.max(max_vec)
        limit = cmax - (cmax - cmin) * threshold
        max_mask = max_vec > limit
        argmax = np.argmax(maps, axis=1)
        return (argmax + 1) * max_mask

    def _multiple_values_max(self, maps, threshold):
        """
        Use a threshold to extract the minimum value along
        the y axis.
        A new threshold value is used for each value of y.
        """
        max_val = np.zeros((maps.shape[0], maps.shape[1]), dtype=np.float)
        for i in range(maps.shape[1]):
            cmin = np.min(maps[:,i])
            cmax = np.max(maps[:,i])
            limit = cmax - (cmax - cmin) * threshold[i]
            min_mask = maps[:,i] <= limit
            max_mask = maps[:,i] > limit
            # for an abundance map the delta is around [-1..1],
            # but it can be outside this interval, it's something
            # to test
            # a guard with a -10 value maybe ok.
            rmin = min_mask * -10
            max_val[:,i] = max_mask * maps[:,i] + rmin
        max_vec = np.max(max_val, axis=1)
        max_mask = max_vec > -10
        argmax = np.argmax(max_val, axis=1)
        return (argmax + 1) * max_mask

    @PlotInputValidation('AbundanceClassification')
    @Plot
    def plot(self, path, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

    @DisplayInputValidation('AbundanceClassification')
    @Display
    def display(self, labels=None, mask=None, interpolation='none', colorMap='Accent', suffix=None):
        pass

import sys
if sys.version_info[0] == 2:
    AbundanceClassification.plot.__func__.__doc__ = plot_docstring
    AbundanceClassification.display.__func__.__doc__ = display_docstring
if sys.version_info[0] == 3:
    AbundanceClassification.plot.__doc__ = plot_docstring
    AbundanceClassification.display.__doc__ = display_docstring

