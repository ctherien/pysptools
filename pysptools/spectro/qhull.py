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
# qhull.py - This file is part of the PySptools package.
#

"""
FeaturesConvexHullQuotient, SpectrumConvexHullQuotient classes
"""

from __future__ import division
from __future__ import print_function

from . import hull_removal as hr
#import .hull_removal as hr
from .inval import *

class SpectrumConvexHullQuotient(object):
    """
    Remove the convex-hull of the signal by hull quotient.

    Parameters:
        spectrum: `list`
            1D HSI data (p), a spectrum.
        wvl: `list`
            Wavelength of each band (p x 1).

    Reference:
        Clark, R.N. and T.L. Roush (1984) Reflectance Spectroscopy: Quantitative
        Analysis Techniques for Remote Sensing Applications, J. Geophys. Res., 89,
        6329-6340.
    """

    @SCHQInitInputValidation('SpectrumConvexHullQuotient')
    def __init__(self, spectrum, wvl, normalize=False):
        import pysptools.util as util
        if normalize == True:
            self.spectrum = util.normalize(spectrum)
        else:
            self.spectrum = spectrum
        self.wvl = wvl
        # continuum removed spectrum
        self.crs = None
        # hull by x
        self.hx = None
        # hull by y
        self.hy = None
        self._remove()

    def get_continuum_removed_spectrum(self):
        """
        Returns: `list`
            Spectrum with convex hull removed (p).
        """
        return self.crs

    def get_hull_x(self):
        """
        Returns: `list`
            Convex hull x values (p).
        """
        return self.hx

    def get_hull_y(self):
        """
        Returns: `list`
            Convex hull y values (p).
        """
        return self.hy

    def _remove(self):
        self.crs, self.hx, self.hy = hr.convex_hull_removal(self.spectrum, self.wvl)

    @PlotInputValidation('SpectrumConvexHullQuotient')
    def plot(self, path, plot_name, suffix=None):
        """
        Plot the hull quotient graph using matplotlib.

        Parameters:
            path: `string`
              The path where to put the plot.

            plot_name: `string`
              File name.

            suffix: `string`
              Add a suffix to the file name.
        """
        import os.path as osp
        import matplotlib.pyplot as plt
        plt.ioff()
        if suffix == None:
            fout = osp.join(path, plot_name + '.png')
        else:
            fout = osp.join(path, plot_name + '_{0}.png'.format(suffix))
        plt.xlabel('Wavelength')
        plt.ylabel('Brightness')
        plt.title('{0} Hull Quotient'.format(plot_name))
        plt.grid(True)
        plt.plot(self.wvl, self.crs, 'g', label='crs')
        plt.plot(self.hx, self.hy, 'c', label='hull')
        plt.plot(self.hx, self.hy, 'r.', label='hull pts')
        plt.plot(self.wvl, self.spectrum, 'b', label='signal')
        plt.legend(framealpha=0.5)
        plt.savefig(fout)
        plt.close()

    @DisplayInputValidation('SpectrumConvexHullQuotient')
    def display(self, plot_name, suffix=None):
        """
        Display the hull quotient graph to the IPython
        Notebook using matplotlib.

        Parameters:
            plot_name: `string`
              File name.

            suffix: `string`
              Add a suffix to the title.
        """
        import matplotlib.pyplot as plt
        plt.xlabel('Wavelength')
        plt.ylabel('Brightness')
        plt.title('{0} Hull Quotient'.format(plot_name))
        plt.grid(True)
        plt.plot(self.wvl, self.crs, 'g', label='crs')
        plt.plot(self.hx, self.hy, 'c', label='hull')
        plt.plot(self.hx, self.hy, 'r.', label='hull pts')
        plt.plot(self.wvl, self.spectrum, 'b', label='signal')
        plt.legend(framealpha=0.5)
        plt.show()
        plt.close()


class FeaturesConvexHullQuotient(SpectrumConvexHullQuotient):
    """
    Remove the convex-hull of the signal by hull quotient.
    Auto-extract the features and calculate their associated
    statistics. A baseline can be applied to avoid non-significant features.
    If you want to restrict the analysis to one continuum, just set the intervale
    with the startContinuum and stopContinuum parameters. It is up to you
    to ascertain that the continuum interval defined by startContinuum and
    stopContinuum do not cross the spectrum. The bilateral function can be use
    to remove small spectrum noises before extracting the features.

    Parameters:
        spectrum: `list`
            1D HSI data (p), a spectrum.

        wvl: `list`
            Wavelength of each band (p x 1).

        startContinuum: `float`
            The wavelength value of the starting left continuum.

        stopContinuum: `float`
            The wavelength value of the ending right continuum.

        baseline: `float`
            Features extracted above the baseline are rejected,
            features extracted below the baseline are kept.

    Reference:
        Kokaly F. Raymond, PRISM: Processing Routines in IDL for Spectroscopic
        Measurements (Installation Manual and User's Guide, Version 1.0),
        U.S. Geological Survey,Reston, Virginia: 2011.
    """

    def __init__(self, spectrum, wvl, startContinuum=None, stopContinuum=None, baseline=0, normalize=False):
        if startContinuum != None and stopContinuum != None:
            start = 0
            for i in range(len(wvl)):
                if wvl[i] > startContinuum:
                    start = i
                    break
            stop = 0
            for i in range(len(wvl)):
                if wvl[i] > stopContinuum:
                    stop = i
                    break
            spectrum = spectrum[start:stop]
            wvl = wvl[start:stop]

        SpectrumConvexHullQuotient.__init__(self, spectrum, wvl, normalize)
        self.base_line = baseline
        self.features = []
        self.features_all = []
        self._extract_features()
        self._base_line_clean()

    def _all_features_number(self):
        return len(self.hx)

    def _features_number(self):
        return len(self.features)

    def _extract_features(self):
        for feat_no in range(self._all_features_number() - 1):
            start1 = self.hx[feat_no]
            end1 = self.hx[feat_no + 1]
            for i in range(len(self.wvl)):
                if start1 == self.wvl[i]: start2 = i
                if end1 == self.wvl[i]:
                    end2 = i
                    break
            spectrum = self.spectrum[start2:end2+1]
            wvl = self.wvl[start2:end2+1]
            crs = self.crs[start2:end2+1]
            hx = self.hx[feat_no:feat_no+2]
            hy = self.hy[feat_no:feat_no+2]
            feat = {'seq': feat_no,
                    'id': None,
                    'state': None,
                    'spectrum': spectrum,
                    'wvl': wvl,
                    'crs': crs,
                    'hx': hx,
                    'hy': hy,
                    'cstart_wvl': None,
                    'cstop_wvl': None,
                    'abs_wvl': None,
                    'abs_depth': None,
                    'area': None,
                    'cslope': None,
                    'FWHM_x': None,
                    'FWHM_y': None,
                    'FWHM_delta': None
                    }
            self.features_all.append(feat)

    def _area(self, y):
        from scipy.integrate import trapz
        # Before the integration:
        # flip the crs curve to x axis
        # and start at y=0
        yy = [abs(p-1) for p in y]
        deltax = self.wvl[1] - self.wvl[0]
        area = trapz(yy, dx=deltax)
        return area

    def _FWHM(self, feat):
        # full_width_at_half_maximum
        import numpy as np
        # get the middle curve value -> y
        depth = np.min(feat['crs'])
        y = depth + ((1 - depth) / 2)
        left_wvl = 0
        # curve_centre is x at the minimum of the curve
        curve_centre = feat['wvl'].index(feat['abs_wvl'])
        # going from the curve center to left
        for i in range(curve_centre,-1,-1):
            if feat['crs'][i] >= y:
                left_wvl = feat['wvl'][i]
                break
        stop = len(feat['wvl'])
        right_wvl = 0
        # going from the curve center to right
        for i in range(curve_centre, stop):
            if feat['crs'][i] >= y:
                right_wvl = feat['wvl'][i]
                break
        delta = right_wvl - left_wvl
        FWHM_x = (left_wvl, right_wvl)
        FWHM_y = (y, y)
        return FWHM_x, FWHM_y, delta

    def _add_stats(self, feat):
        import numpy as np
        feat['area'] = self._area(feat['crs'])
        feat['cstart_wvl'] = feat['wvl'][0]
        feat['cstop_wvl'] = feat['wvl'][-1]
        feat['abs_wvl'] = feat['wvl'][np.argmin(feat['crs'])]
        feat['abs_depth'] = np.min(feat['crs'])
        feat['cslope'] = (feat['hy'][1] - feat['hy'][0]) / (feat['hx'][1] - feat['hx'][0])
        feat['FWHM_x'], feat['FWHM_y'], feat['FWHM_delta'] = self._FWHM(feat)

    def _base_line_clean(self):
        id = 1
        for feat in self.features_all:
            if min(feat['crs']) < self.base_line:
                feat['state'] = 'keep'
                feat['id'] = id
                id += 1
                self._add_stats(feat)
                self.features.append(feat)
            else:
                feat['state'] = 'reject'

    def get_number_of_kept_features(self):
        """
        Returns: `int`
            The number of features that are kept (below the baseline).
            Only theses features have a statistic report.
        """
        return self._features_number()

    def get_continuum_removed_spectrum(self, feat_no):
        """
        Returns: `list`
            Feature spectrum with convex hull removed (p).
        """
        return self.features[feat_no-1]['crs']

    def get_absorbtion_wavelength(self, feat_no):
        """
        Returns: `float`
            The wavelength at the feature minimum.
        """
        return self.features[feat_no-1]['abs_wvl']

    def get_absorbtion_depth(self, feat_no):
        """
        Returns: `float`
            The absorbtion value at the feature minimum.
        """
        return self.features[feat_no-1]['abs_depth']

    def get_continuum_slope(self, feat_no):
        """
        Returns: `float`
            The feature continuum slope.
        """
        return self.features[feat_no-1]['cslope']

    def get_area(self, feat_no):
        """
        Returns: `float`
            The feature area.
        """
        return self.features[feat_no-1]['area']

    def get_continuum_start_wavelength(self, feat_no):
        """
        Returns: `float`
            The continuum left start wavelength value.
        """
        return self.features[feat_no-1]['cstart_wvl']

    def get_continuum_stop_wavelength(self, feat_no):
        """
        Returns: `float`
            The continuum right end wavelength value.
        """
        return self.features[feat_no-1]['cstop_wvl']

    def get_full_width_at_half_maximum(self, feat_no):
        """
        Returns: `float`
            Width at half maximum.
        """
        return self.features[feat_no-1]['FWHM_delta']

    def print_stats(self, feat_no):
        """Print a statistic summary for a kept feature.

            Parameters:
                feat_no: `int or 'all'`
                  The feature number, if feat_no='all', print
                  stats for all the kept features.
        """
        if feat_no == 'all':
            for i in range(len(self.features)):
                self._print_stats1(self.features[i])
        else:
            feat = self.features[feat_no-1]
            self._print_stats1(feat)

    def _print_stats1(self, feat):
        print('Feature Stats')
        print('  ---------------------------')
        print('  feature number:',feat['id'])
        print('  ---------------------------')
        print('  area:',feat['area'])
        print('  ---------------------------')
        print('  continuum start wavelength:',feat['cstart_wvl'])
        print('  continuum stop wavelength:',feat['cstop_wvl'])
        print('  continuum slope:',feat['cslope'])
        print('  ---------------------------')
        print('  center wavelength:',feat['abs_wvl'])
        print('  depth:',feat['abs_depth'])
        print('  ---------------------------')
        print('  full-width at half maximum:',feat['FWHM_delta'])
        print()

    @PlotInputValidation2('FeaturesConvexHullQuotient')
    def plot(self, path, plot_name, feature='all'):
        """
        Plot the hull quotient graph of the feature using matplotlib.

        Parameters:
            path: `string`
              The path where to put the plot.

            plot_name: `string`
              File name.

            suffix: `string`
              Add a suffix to the file name.
        """
        if feature == 'all':
            for i in range(self._features_number()):
                self._feature_plot1(path, plot_name, self.features[i])
        else:
            self._feature_plot1(path, plot_name, self.features[feature-1])

    def _feature_plot1(self, path, plot_name, feat):
            import os.path as osp
            import matplotlib.pyplot as plt
            plt.ioff()
            fout = osp.join(path, plot_name + '_feature {0}'.format(feat['id']) + '.png')
            plt.xlabel('Wavelength')
            plt.ylabel('Brightness')
            plt.title('{0} Hull Quotient, feature {1}'.format(plot_name, feat['id']))
            plt.grid(True)
            plt.plot(feat['wvl'], feat['crs'], 'g', label='crs')
            x = (feat['abs_wvl'],feat['abs_wvl'])
            y = (feat['abs_depth'],1)
            plt.plot(x, y, 'r', label='depth')
            plt.plot(feat['FWHM_x'], feat['FWHM_y'], 'b', label='FWHM')
            plt.legend(framealpha=0.5)
            plt.savefig(fout)
            plt.clf()

    @DisplayInputValidation2('FeaturesConvexHullQuotient')
    def display(self, plot_name, feature='all'):
        """
        Display the hull quotient graph of the feature
        to the IPython Notebook using matplotlib.

        Parameters:
            plot_name: `string`
              Title name.

            suffix: `string`
              Add a suffix to the title.
        """
        if feature == 'all':
            for i in range(self._features_number()):
                self._feature_display1(plot_name, self.features[i])
        else:
            self._feature_display1(plot_name, self.features[feature-1])

    def _feature_display1(self, plot_name, feat):
            import matplotlib.pyplot as plt
            plt.xlabel('Wavelength')
            plt.ylabel('Brightness')
            plt.title('{0} Hull Quotient, feature {1}'.format(plot_name, feat['id']))
            plt.grid(True)
            plt.plot(feat['wvl'], feat['crs'], 'g', label='crs')
            x = (feat['abs_wvl'],feat['abs_wvl'])
            y = (feat['abs_depth'],1)
            plt.plot(x, y, 'r', label='depth')
            plt.plot(feat['FWHM_x'], feat['FWHM_y'], 'b', label='FWHM')
            plt.legend(framealpha=0.5)
            plt.show()
            plt.clf()

    def plot_convex_hull_quotient(self, path, plot_name, suffix=None):
        """
        Plot the hull quotient graph using matplotlib.

        Parameters:
            path: `string`
              The path where to put the plot.

            plot_name: `string`
              File name.

            suffix: `string`
              Add a suffix to the file name.
        """
        SpectrumConvexHullQuotient.plot(self, path, plot_name, suffix)

    def display_convex_hull_quotient(self, plot_name, suffix=None):
        """
        Display the hull quotient graph to the IPython
        Notebook using matplotlib.

        Parameters:
            plot_name: `string`
              Title name.

            suffix: `string`
              Add a suffix to the title.
        """
        SpectrumConvexHullQuotient.display(self, plot_name, suffix)

