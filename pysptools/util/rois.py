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
# rois.py - This file is part of the PySptools package.
#

from __future__ import print_function

import numpy as np

class ROIs(object):
    """
    Manage regions of interest (ROIs).
    """

    def __init__(self, x, y):
        """
        Support regions of interest.

        Parameters:
            x: `int`
              The cube horizontal dimension (M.shape[0]).

            y: `int`
              The cube vertical dimension (M.shape[1]).
        """
        self._rois = []
        self._n_clusters = 0
        # mask: a value of zero is X, and a value of 1,2... is y, 1 for the first
        # roi, 2 for the next roi ...
        self.mask = np.zeros((x, y), dtype=np.int)

    def add(self, id, *rois):
        """
        Add a named ROI.

        Parameters:
            id: `string`
              The class (or cluster) name.

            rois: `dictionary list`
              Each parameter, a dictionary, represent a rectangle, a polygon or a raw array. They use matrix coordinates.
              For a raw array: {'raw': mask-array}, mask-array is a binary 2D array with the hypercube (x,y) dimensions.
              For a rectangle: {'rec': (upper_left_line, upper_left_column, lower_right_line, lower_right_column)}.
              For a polygone: {'poly': ((l1,c1),(l2,c2), ...)}, **l** stand for line and **c** for column. The polygon don't need to be close.
              You can define one or more raw, rectangle and/or polygon for a same cluster.
              The polygon and the rectangle must be well formed.
        """
        self._rois.append((id, rois))
        self._n_clusters += 1
        self._post_to_mask(rois, self._n_clusters)

    def n_clusters(self):
        return self._n_clusters

    def get_mask(self):
        return self.mask

    def get_next(self):
        """
        Iterator, return at each step: the cluster name and a ROI list.

        Return: `tuple`
            Cluster name, ROI list.
        """
        for r in self._rois:
            id_ = r[0]
            rois = r[1]
            yield id_, rois

    def get_labels(self):
        """
        Return a labels list.

        Return: `list`
            A labels list.
        """
        labels = []
        for r in self._rois:
            labels.append(r[0])
        return labels

    def _post_to_mask(self, rois, id):
        for r in rois:
            if 'raw' in r:
                bin_mask = r['raw']
                for x in range(self.mask.shape[0]):
                    for y in range(self.mask.shape[1]):
                        if bin_mask[x,y] > 0:
                            self.mask[x,y] = id
            if 'rec' in r:
                x1,y1,x2,y2 = r['rec']
                for x in range(self.mask.shape[0]):
                    for y in range(self.mask.shape[1]):
                        if (x >= x1 and x < x2) and (y >= y1 and y < y2):
                            self.mask[x,y] = id
            if 'poly' in r:
                import matplotlib.patches as patches
                poly1 = patches.Polygon(r['poly'], closed=True)
                for i in range(self.mask.shape[0]):
                    for j in range(self.mask.shape[1]):
                        if poly1.get_path().contains_point((i,j)) == True:
                            self.mask[i,j] = id

    def plot(self, path, colorMap='Accent', suffix=None):
        """
        Plot the ROIs.

        Parameters:
            path: `string`
              The path where to put the plot.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        from pysptools.classification import Output
        plt = Output('ROIs')
        plt.plot(self.mask, self._n_clusters, path=path, labels=self.get_labels(), colorMap=colorMap, firstBlack=True, suffix=suffix)

    def display(self, colorMap='Accent', suffix=None):
        """
        Display the ROIs.

        Parameters:
            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        from pysptools.classification import Output
        plt = Output('ROIs')
        plt.plot(self.mask, self._n_clusters, labels=self.get_labels(), colorMap=colorMap, firstBlack=True, suffix=suffix)

if __name__ == '__main__':
    pass