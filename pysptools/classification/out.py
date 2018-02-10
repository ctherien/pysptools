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
# out.py - This file is part of the PySptools package.
#

import os.path as osp
import numpy as np


class Output(object):
    """ Add plot and display functionality to the classifiers classes.
    """

    def __init__(self, label):
        self.label = label

    def cm_dispatch(self, name):
        from pysptools.classification._cm import datad
        return datad[name]
    
    def _custom_listed_color_map(self, name, N, firstBlack=False):
        """ add the black color in front of 'name' color """
        import matplotlib.cm as cm
        from matplotlib import colors
        if name == 'jet':
            mp = cm.datad[name]
        else:
            mp = self.cm_dispatch(name)            
        new_mp1 = {'blue': colors.makeMappingArray(N-1, mp['blue']),
                  'green': colors.makeMappingArray(N-1, mp['green']),
                  'red': colors.makeMappingArray(N-1, mp['red'])}
        new_mp2 = []
        new_mp2.extend(zip(new_mp1['red'], new_mp1['green'], new_mp1['blue']))
        if firstBlack == True:
            new_mp2 = [(0,0,0)]+new_mp2 # the black color
        return colors.ListedColormap(new_mp2, N=N-1), new_mp2

    def plot(self, img, n_classes, path=None, labels=None, mask=None, interpolation='none', colorMap='jet', firstBlack=False, suffix=''):
        """
        Plot a classification map using matplotlib.

        Parameters:
            img: `numpy array`
                A classified map, (m x n x 1),
                the classes start at 0.

            n_classes: `int`
                The number of classes found in img.

            path: `string`
                The path where to put the plot.

            labels: `list of string [default None]`
                The legend labels.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding pixel is displayed.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            firstBlack: `bool [default False]`
                Display the first legend element in black if *True*. If it is the case,
                the corresponding classification class value is zero and it can be use when the
                meaning is nothing to classify (example: a background).

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import colors
        if path != None:
            plt.ioff()
        # fallback on jet colormap
        if colorMap == 'Accent': color = cm.Accent
        elif colorMap == 'Dark2': color = cm.Dark2
        elif colorMap == 'Paired': color = cm.Paired
        elif colorMap == 'Pastel1': color = cm.Pastel1
        elif colorMap == 'Pastel2': color = cm.Pastel2
        elif colorMap == 'Set1': color = cm.Set1
        elif colorMap == 'Set2': color = cm.Set2
        elif colorMap == 'Set3': color = cm.Set3
        else:
            color = cm.jet
            colorMap = 'jet'

        if isinstance(mask, np.ndarray):
            img = img[:,:] * mask

        if firstBlack == False: n_classes = n_classes - 1
        bounds = range(n_classes+2)
        color, dummy = self._custom_listed_color_map(colorMap, len(bounds)+1, firstBlack=firstBlack)
        norm = colors.BoundaryNorm(bounds, color.N)
        fig0, ax0 = plt.subplots()
        plot = ax0.imshow(img, cmap=color, interpolation=interpolation, norm=norm)
        cbar = fig0.colorbar(plot, cmap=color, norm=norm, boundaries=bounds,
                           ticks=[x+0.5 for x in range(n_classes+1)])

        if labels == None:
            if firstBlack == True:
                sigSet = [x+1 for x in range(n_classes)]
                lbls = ['None']
                lbls.extend(sigSet)
            else:
                sigSet = [x+1 for x in range(n_classes+1)]
                lbls = []
                lbls.extend(sigSet)
        else:
            if firstBlack == True:
                lbls = ['None']
                lbls.extend(labels)
            else:
                lbls = []
                lbls.extend(labels)
        cbar.set_ticklabels(lbls)
        
        if labels == None:
            ax0.set_ylabel('class #', rotation=270, labelpad=70)
            ax0.get_yaxis().set_label_position("right")

        if path != None:
            if suffix == None:
                fout = osp.join(path, '{0}.png'.format(self.label))
            else:
                fout = osp.join(path, '{0}_{1}.png'.format(self.label, suffix))
            try:
                plt.savefig(fout)
            except IOError:
                raise IOError('in classification.SVC, no such file or directory: {0}'.format(path))
        else:
            if suffix == None:
                plt.title('{0}'.format(self.label))
            else:
                plt.title('{0} - {1}'.format(self.label, suffix))
            plt.show()
        plt.close()

    def display(self, img, n_classes, labels=None, mask=None, interpolation='none', colorMap='jet', firstBlack=False, suffix=''):
        """
        Display a classification map using matplotlib.

        Parameters:
            img: `numpy array`
                A classified map, (m x n x 1),
                the classes start at 0.

            n_classes: `int`
                The number of classes found in img.

            labels: `list of string [default None]`
                The legend labels.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding pixel is displayed.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            firstBlack: `bool [default False]`
                Display the first legend element in black if *True*. If it is the case,
                the corresponding classification class value is zero and it can be use when the
                meaning is nothing to classify (example: a background).

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        self.plot(img, n_classes, labels=labels, mask=mask, interpolation=interpolation, colorMap=colorMap, firstBlack=firstBlack, suffix=suffix)

    def plot1(self, img, path=None, mask=None, interpolation='none', colorMap='jet', suffix=''):
        import matplotlib.pyplot as plt
        if path != None:
            plt.ioff()

        if isinstance(mask, np.ndarray):
            img = img[:,:] * mask

        plt.imshow(img, interpolation=interpolation)
        plt.set_cmap(colorMap)
        cbar = plt.colorbar()
        cbar.set_ticks([])
        if path != None:
            if suffix == None:
                fout = osp.join(path, '{0}.png'.format(self.label))
            else:
                fout = osp.join(path, '{0}_{1}.png'.format(self.label, suffix))
            try:
                plt.savefig(fout)
            except IOError:
                raise IOError('in classifiers.output, no such file or directory: {0}'.format(path))
        else:
            if suffix == None:
                plt.title('{0}'.format(self.label))
            else:
                plt.title('{0} - {1}'.format(self.label, suffix))
            plt.show()

        plt.close()

    def plot_single_map(self, path, cmap, dist_map, lib_idx, em_nbr, threshold, constrained, stretch, colorMap, suffix):
##        """
##        Plot individual classified map. One for each spectrum.
##
##        Parameters
##            path : string
##              The path where to put the plot.
##
##            lib_idx : int
##                * A number between 1 and the number of spectra in the library.
##                * 'all', plot all the individual maps.
##
##            suffix : string
##              Add a suffix to the file name.
##
##        """
        if lib_idx == 'all':
            for signo in range(em_nbr):
                self._plot_single_map1(path, cmap, signo + 1, dist_map, threshold, constrained, stretch, colorMap, suffix)
        else:
            self._plot_single_map1(path, cmap, lib_idx, dist_map, threshold, constrained, stretch, colorMap, suffix)

    def _plot_single_map1(self, path, cmap, signo, dist_map, threshold, constrained, stretch, colorMap, suffix):
        import matplotlib.pyplot as plt
        if path != None:
            plt.ioff()
        grad = self.get_single_map(signo, cmap, dist_map, threshold, constrained, stretch)
        plt.imshow(grad, interpolation='none')
        plt.set_cmap(colorMap)
        cbar = plt.colorbar()
        cbar.set_ticks([])
        if path != None:
            if suffix == None:
                fout = osp.join(path, '{0}_{1}.png'.format(self.label, signo))
            else:
                fout = osp.join(path, '{0}_{1}_{2}.png'.format(self.label, signo, suffix))
            try:
                plt.savefig(fout)
            except IOError:
                raise IOError('in classifiers.output, no such file or directory: {0}'.format(path))
        else:
            if suffix == None:
                plt.title('{0} - EM{1}'.format(self.label, signo))
            else:
                plt.title('{0} - EM{1} - {2}'.format(self.label, signo, suffix))
            plt.show()
        plt.close()

    def get_single_map(self, signo, cmap, dist_map, threshold, constrained, stretch, inverse_scale=True):
        if constrained == False:
            amin = np.min(dist_map[:,:,signo - 1])
            amax = np.max(dist_map[:,:,signo - 1])
            if type(threshold) is float:
                limit = amin + (amax - amin) * threshold
            if type(threshold) is list:
                limit = amin + (amax - amin) * threshold[signo - 1]
            if self.label == 'NormXCorr':
                grad = (dist_map[:,:,signo - 1] > limit) * dist_map[:,:,signo - 1]
            else:
                grad = (dist_map[:,:,signo - 1] < limit) * dist_map[:,:,signo - 1]
        if constrained == True:
            thresholded = cmap == signo
            grad = (dist_map[:,:,signo - 1] * thresholded)
        # Inverse the scale for SAM and SID,
        # not needed for NormXCorr
        # Some brain damaging logic here:
        # inverse_scale == True only for the plot_single_map() call
        # inverse_scale == False only for the get_single_map() call
        if inverse_scale == True:
            if self.label == 'NormXCorr' and stretch == True:
                # strech between 0 and 1
                min = 2
                for i in range(grad.shape[0]):
                    for j in range(grad.shape[1]):
                        if grad[i,j] < min and grad[i,j] != 0:
                            min = grad[i,j]
                for i in range(grad.shape[0]):
                    for j in range(grad.shape[1]):
                        if grad[i,j] != 0:
                            grad[i,j] = grad[i,j] - min
                grad = (1 / np.max(grad)) * grad
            if self.label == 'SAM' or self.label == 'SID':
                # strech between 0 and 1
                if stretch == True:
                    grad = (1 / np.max(grad)) * grad
                # and inverse
                for i in range(grad.shape[0]):
                    for j in range(grad.shape[1]):
                        if grad[i,j] != 0: grad[i,j] = 1 - grad[i,j]
        return grad

    def plot_histo(self, path, cmap, em_nbr, suffix):
        import matplotlib.pyplot as plt
        plt.ioff()
        farray = np.ndarray.flatten(cmap)
        plt.hist(farray, bins=range(em_nbr+2), align='left')
        if suffix == None:
            fout = osp.join(path, 'histo_{0}.png'.format(self.label))
        else:
            fout = osp.join(path, 'histo_{0}_{1}.png'.format(self.label, suffix))
        try:
            plt.savefig(fout)
        except IOError:
            raise IOError('in classifiers.output, no such file or directory: {0}'.format(path))
        plt.close()
