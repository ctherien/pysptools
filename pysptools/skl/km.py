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
# km.py - This file is part of the PySptools package.
#

"""
KMeans class
"""

import numpy as np
import sklearn.cluster as cluster
#from . import out
#from .inval import *
from pysptools.classification.out import Output
from pysptools.classification.inval import *


class KMeans(object):
    """ KMeans clustering algorithm adapted to hyperspectral imaging """

    def __init__(self):
        self.cluster = None
        self.n_clusters = None
        self.output = Output('KMeans')

    @PredictInputValidation('KMeans')
    def predict(self, M, n_clusters=5, n_jobs=1, init='k-means++'):
        """
        KMeans clustering algorithm adapted to hyperspectral imaging.
        It is a simple wrapper to the scikit-learn version.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

            n_clusters: `int [default 5]`
                The number of clusters to generate.

            n_jobs: `int [default 1]`
                Taken from scikit-learn doc:
                The number of jobs to use for the computation. This works by breaking down the pairwise matrix into n_jobs even slices and computing them in parallel.
                If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.

            init: `string or array [default 'k-means++']`
                Taken from scikit-learn doc: Method for initialization, defaults to `k-means++`:
                `k-means++` : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
                `random`: choose k observations (rows) at random from data for the initial centroids.
                If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.


        Returns: `numpy array`
              A cluster map (m x n x c), c is the clusters number .

        """
        h, w, numBands = M.shape
        self.n_clusters = n_clusters
        X = np.reshape(M, (w*h, numBands))
        clf = cluster.KMeans(n_clusters=n_clusters, n_jobs=n_jobs, init=init)
        cls = clf.fit_predict(X)
        self.cluster = np.reshape(cls, (h, w))
        return self.cluster

    @PlotInputValidation3('KMeans')
    def plot(self, path, interpolation='none', colorMap='Accent', suffix=None):
        """
        Plot the cluster map.

        Parameters:
            path: `string`
              The path where to put the plot.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        self.output.plot(self.cluster, self.n_clusters, path=path, interpolation=interpolation, colorMap=colorMap, suffix=suffix)

    @DisplayInputValidation3('KMeans')
    def display(self, interpolation='none', colorMap='Accent', suffix=None):
        """
        Display the cluster map.

        Parameters:
            path: `string`
              The path where to put the plot.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the title.
        """
        self.output.plot(self.cluster, self.n_clusters, interpolation=interpolation, colorMap=colorMap, suffix=suffix)

