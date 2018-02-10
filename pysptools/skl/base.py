#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2017, Christian Therien
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
# base.py - This file is part of the PySptools package.
#

import numpy as np
from sklearn import preprocessing
from pysptools.classification.out import Output
from pysptools.classification.inval import *


class HyperBaseClassifier(object):
    
    cmap = None

    def __init__(self, label):
        self.output = Output(label)

    def _fit_rois(self, M, ROIs):
        self.n_clusters = ROIs.n_clusters()
        self.mask = ROIs.get_mask()
        X_cube = self._get_X(M)

        # a value of 0 for 'y' represents the background cluster
        y_cube = np.zeros(X_cube.shape[0], dtype=np.int)
        X = np.array(X_cube)
        y = np.array(y_cube)

        # i = 1 for the first cluster, i = 2 for the second cluster ...
        i = 0
        for id_, rois in ROIs.get_next():
            i += 1
            for r in rois:
                X_roi = self._crop(M, i, r)
                y_roi = np.zeros(X_roi.shape[0], dtype=np.int) + i
                X = np.concatenate((X, X_roi))
                y = np.concatenate((y, y_roi))
        return X, y

    def get_class_map(self):
        return self.cmap

    def _crop(self, M, roi_id,  r):
        if 'rec' in r:
            bbox = r['rec']
            return self._convert2D(M[bbox[0]:bbox[2],bbox[1]:bbox[3],:])
        if 'poly' in r or 'raw' in r:
            masked = np.sum(self.mask == roi_id)
            linear_cube = np.ndarray((masked, M.shape[2]), dtype=np.float)
            i = 0
            for x in range(M.shape[0]):
                for y in range(M.shape[1]):
                    if self.mask[x,y] == roi_id:
                        linear_cube[i] = M[x,y,:]
                        i += 1
            return linear_cube

    def _convert2D(self, M):
        h, w, numBands = M.shape
        return np.reshape(M, (w*h, numBands))

    # Convert a cmap to 3D
    def _convert3d(self, M, h, w):
        return np.reshape(M, (h, w))

    def _get_X(self, M):
        masked = np.sum(self.mask > 0)
        not_masked = M.shape[0] * M.shape[1] - masked
        linear_cube = np.ndarray((not_masked, M.shape[2]), dtype=np.float)
        i = 0
        for x in range(M.shape[0]):
            for y in range(M.shape[1]):
                if self.mask[x,y] == 0:
                    linear_cube[i] = M[x,y,:]
                    i += 1
        return linear_cube

    def _set_cmap(self, m):
        self.cmap = m

    def _set_n_clusters(self, m):
        self.n_clusters = m

    @PlotInputValidation2('HyperBaseClassifier')
    def plot(self, path, labels=None, interpolation='none', colorMap='Accent', suffix=None):
        self.output.plot(self.cmap, self.n_clusters, path=path, labels=labels, interpolation=interpolation, colorMap=colorMap, firstBlack=True, suffix=suffix)

    @DisplayInputValidation2('HyperBaseClassifier')
    def display(self, labels=None, interpolation='none', colorMap='Accent', suffix=None):
        self.output.plot(self.cmap, self.n_clusters, labels=labels, interpolation=interpolation, colorMap=colorMap, firstBlack=True, suffix=suffix)


class HyperScaledBaseClassifier(HyperBaseClassifier):

    def _fit_rois(self, M, ROIs):
        self.n_clusters = ROIs.n_clusters()
        self.mask = ROIs.get_mask()
        X_cube = self._get_X(M)

        # a value of 0 for 'y' represents the background cluster
        y_cube = np.zeros(X_cube.shape[0], dtype=np.int)
        X = np.array(X_cube)
        y = np.array(y_cube)

        # i = 1 for the first cluster, i = 2 for the second cluster ...
        i = 0
        for id_, rois in ROIs.get_next():
            i += 1
            for r in rois:
                X_roi = self._crop(M, i, r)
                y_roi = np.zeros(X_roi.shape[0], dtype=np.int) + i
                X = np.concatenate((X, X_roi))
                y = np.concatenate((y, y_roi))
        X_scaled = preprocessing.scale(X)
        return X_scaled, y
