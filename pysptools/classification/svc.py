#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2015, Christian Therien
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
# svc.py - This file is part of the PySptools package.
#


import numpy as np
from sklearn import svm
from sklearn import preprocessing
from . import out
from .inval import *


class SVC(object):
    """
    Suppot Vector Supervised Classification (SVC) of a HSI cube with the
    use of regions of interest (ROIs).

    This class is largely a wrapper to the scikit-learn SVC class. The goal is
    to ease the use of the scikit-learn SVM implementation when applied
    to hyperspectral cubes.

    The ROIs classifiers can be rectangles or polygons. They must be VALID, no check is made
    upon the validity of these geometric figures.
    """

    def __init__(self):
        self.clf = None
        self.cmap = None
        self.mask = None
        self.n_clusters = None
        self.output = out.Output('SVC')

    def fit(self, M, ROIs, class_weight=None, cache_size=200, coef0=0.0, degree=3,
            gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None,
            shrinking=True, tol=0.001, verbose=False):
        """
        Fit the HS cube M with the use of ROIs. The parameters following 'M' and 'ROIs' are the
        one defined by the scikit-learn sklearn.svm.SVC class.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

            ROIs: `ROIs type`
                Regions of interest instance.

            Others parameters: `see the sklearn.svm.SVC class parameters`
                Note: the C parameter is set to 1, the result of this setting is that
                the class_weight is relative to C and that the first value of
                class_weight is the background.
                An example: you wish to fit two classes "1" and "2" with the help
                of one ROI for each, you declare class_weight like this:
                class_weight={0:1,1:10,2:10}
                0: is always the background and is set to 1, 1: is the first class,
                2: is the second. A value of 10 for both classes give good results.

        Returns: `class`
            The sklearn.svm.SVC class is returned.
        """
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
        self.clf = svm.SVC(C=1.0, class_weight=class_weight, cache_size=cache_size,
                    coef0=coef0, degree=degree, gamma=gamma, kernel=kernel, max_iter=max_iter,
                    probability=probability, shrinking=shrinking, tol=tol, verbose=verbose)
        self.clf.fit(X_scaled, y)
        return self.clf

    @ClassifyInputValidation3('SVC')
    def classify(self, M):
        """
        Classify a hyperspectral cube using the ROIs defined clusters by the fit method.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

        Returns: `numpy array`
              A class map (m x n x 1).
        """
        img = self._convert2D(M)
        image_scaled = preprocessing.scale(img)
        cls = self.clf.predict(image_scaled)
        self.cmap = self._convert3d(cls, M.shape[0], M.shape[1])
        return self.cmap

    def get_class_map(self):
        return self.cmap

    def _crop(self, M, roi_id,  r):
        if 'rec' in r:
            bbox = r['rec']
            return self._convert2D(M[bbox[0]:bbox[2],bbox[1]:bbox[3],:])
        if 'poly' in r:
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

    @PlotInputValidation2('SVC')
    def plot(self, path, labels=None, interpolation='none', colorMap='Accent', suffix=None):
        """
        Plot the class map.

        Parameters:
            path: `string`
              The path where to put the plot.

            labels: `string list`
              A labels list.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        self.output.plot(self.cmap, self.n_clusters, path=path, labels=labels, interpolation=interpolation, colorMap=colorMap, firstBlack=True, suffix=suffix)

    @DisplayInputValidation2('SVC')
    def display(self, labels=None, interpolation='none', colorMap='Accent', suffix=None):
        """
        Display the class map.

        Parameters:
            labels: `string list`
              A labels list.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        self.output.plot(self.cmap, self.n_clusters, labels=labels, interpolation=interpolation, colorMap=colorMap, firstBlack=True, suffix=suffix)
