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
# dnoise_int.py - This file is part of the PySptools package.
#

from __future__ import division

import os.path as osp
import numpy as np
from scipy.signal import fftconvolve
from math import factorial
from . import dnoise
from .inval import *


class SavitzkyGolay(object):
    """Apply a Savitzky Golay low pass filter."""

    def __init__(self):
        self.denoised = None
        self.dbands = None

    @DenoiseSpectraInputValidation('SavitzkyGolay')
    def denoise_spectra(self, M, window_size, order, deriv=0, rate=1):
        """
        Apply the Savitzky Golay filter on each spectrum.
        Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

            window_size: `int`
                the length of the window. Must be an odd integer number.

            order: `int`
                the order of the polynomial used in the filtering.
                Must be less then `window_size` - 1.

            deriv: `int [default 0]`
                the order of the derivative to compute
                (default = 0 means only smoothing).

        Returns: `numpy array`
              the smoothed signal (or it's n-th derivative) (m x n x p).

        Code source:
            The scipy Cookbook, SavitzkyGolay section. This class is not under the
            copyright of this file.
        """
        h, w, numBands = M.shape
        M = np.reshape(M, (w*h, numBands))
        self.denoised = self._denoise1d(M, window_size, order, deriv, rate)
        self.denoised = np.reshape(self.denoised, (h, w, numBands))
        return self.denoised

    @DenoiseBandsInputValidation('SavitzkyGolay')
    def denoise_bands(self, M, window_size, order, derivative=None):
        """
        Apply the Savitzky Golay filter on each band.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

            window_size: `int`
                the length of the window. Must be an odd integer number.

            order: `int`
                the order of the polynomial used in the filtering.
                Must be less then `window_size` - 1.

            derivative: `string [default None]`
                direction of the derivative to compute, can be None,
                'col', 'row', 'both'.

        Returns: `numpy array`
              the smoothed signal (or it's n-th derivative) (m x n x p).

        Code source:
            The scipy Cookbook, SavitzkyGolay section. This class is not under the
            copyright of this file.
        """
        h, w, numBands = M.shape
        self.dbands = np.ones((h, w, numBands), dtype=np.float)
        for i in range(numBands):
            self.dbands[:,:,i] = self._denoise2d(M[:,:,i], window_size, order, derivative)
        return self.dbands

    def _denoise1d(self, M, window_size, order, deriv, rate):
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError as msg:
            raise ValueError("in SavitzkyGolay.denoise_spectra(), window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("in SavitzkyGolay.denoise_spectra(), window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("in SavitzkyGolay.denoise_spectra(), window_size is too small for the polynomials order")

        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        N, p = M.shape
        dn = np.ones((N,p), dtype=np.float)
        long_signal = np.ndarray(p+2, dtype=np.float)
        for i in range(N):
            y = M[i]
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            long_signal = np.concatenate((firstvals, y, lastvals))
            dn[i] = fftconvolve(long_signal, m, mode='valid')
        return dn

    def _denoise2d(self, z, window_size, order, derivative=None):
        # number of terms in the polynomial expression
        n_terms = ( order + 1 ) * ( order + 2)  / 2.0

        if  window_size % 2 == 0:
            raise ValueError('in SavitzkyGolay.denoise_bands(), window_size must be odd')

        if window_size**2 < n_terms:
            raise ValueError('in SavitzkyGolay.denoise_bands(), order is too high for the window size')

        half_size = window_size // 2

        # exponents of the polynomial.
        # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
        # this line gives a list of two item tuple. Each tuple contains
        # the exponents of the k-th term. First element of tuple is for x
        # second element for y.
        # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
        exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

        # coordinates of points
        ind = np.arange(-half_size, half_size+1, dtype=np.float64)
        dx = np.repeat( ind, window_size )
        dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

        # build matrix of system of equation
        A = np.empty( (window_size**2, len(exps)) )
        for i, exp in enumerate( exps ):
            A[:,i] = (dx**exp[0]) * (dy**exp[1])

        # pad input array with appropriate values at the four borders
        new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
        Z = np.zeros( (new_shape) )
        # top band
        band = z[0, :]
        Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
        # bottom band
        band = z[-1, :]
        Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
        # left band
        band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
        Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
        # right band
        band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
        Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
        # central band
        Z[half_size:-half_size, half_size:-half_size] = z

        # top left corner
        band = z[0,0]
        Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
        # bottom right corner
        band = z[-1,-1]
        Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

        # top right corner
        band = Z[half_size,-half_size:]
        Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
        # bottom left corner
        band = Z[-half_size:,half_size].reshape(-1,1)
        Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

        # solve system and convolve
        if derivative == None:
            m = np.linalg.pinv(A)[0].reshape((window_size, -1))
            return fftconvolve(Z, m, mode='valid')
        elif derivative == 'col':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            return fftconvolve(Z, -c, mode='valid')
        elif derivative == 'row':
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return fftconvolve(Z, -r, mode='valid')
        elif derivative == 'both':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')

    def plot_spectrum_sample(self, M, path, x, y, suffix=None):
        """
        Plot a spectrum sample with the original and the
        filtered signal.

        Parameters:
            M: 'numpy array'
                The original cube (m x n x p).

            path: `string`
              The path where to put the plot.

            x: `int`
                The x coordinate.

            y: `int`
                The y coordinate.

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.plot(M[x,y])
        plt.plot(self.denoised[x,y], color='r')
        plt.xlabel('Wavelength')
        plt.ylabel('Brightness')
        if suffix == None:
            fout = osp.join(path, 'SavitzkyGolay_x{0}_y{1}.png'.format(x,y))
        else:
            fout = osp.join(path, 'SavitzkyGolay_x{0}_y{1}_{2}.png'.format(x,y,suffix))
        plt.savefig(fout)
        plt.close()

    def plot_bands_sample(self, path, band_no, suffix=None):
        """
        Plot a filtered band.

        Parameters:
            path: `string`
              The path where to put the plot.

            band_no: `int or string`
                The band index.
                If band_no == 'all', plot all the bands.

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        import matplotlib.pyplot as plt
        plt.ioff()
        if band_no == 'all':
            for i in range(self.dbands.shape[2]):
                plt.imshow(self.dbands[:,:,i], interpolation='none')
                if suffix == None:
                    fout = osp.join(path, 'SavitzkyGolay_band_{0}.png'.format(i))
                else:
                    fout = osp.join(path, 'SavitzkyGolay_band_{0}_{1}.png'.format(i, suffix))
                plt.savefig(fout)
                plt.close()
        else:
            plt.imshow(self.dbands[:,:,band_no], interpolation='none')
            if suffix == None:
                fout = osp.join(path, 'SavitzkyGolay_band_{0}.png'.format(band_no))
            else:
                fout = osp.join(path, 'SavitzkyGolay_band_{0}_{1}.png'.format(band_no, suffix))
            plt.savefig(fout)
            plt.close()

    def display_spectrum_sample(self, M, x, y, suffix=None):
        """
        Display a spectrum sample with the original and the
        filtered signal.

        Parameters:
            M: 'numpy array'
                The original cube (m x n x p).

            x: `int`
                The x coordinate.

            y: `int`
                The y coordinate.

            suffix: `string [default None]`
              Add a suffix to the title.
        """
        import matplotlib.pyplot as plt
        plt.plot(M[x,y])
        plt.plot(self.denoised[x,y], color='r')
        plt.xlabel('Wavelength')
        plt.ylabel('Brightness')
        if suffix == None:
            plt.title('SG spectrum sample, x={0}, y={1}'.format(x,y))
        else:
            plt.title('SG spectrum sample, x={0}, y={1} - {2}'.format(x,y,suffix))
        plt.show()
        plt.close()

    def display_bands_sample(self, band_no, suffix=None):
        """
        Display a filtered band.

        Parameters:
            band_no: `int or string`
                The band index.
                If band_no == 'all', plot all the bands.

            suffix: `string [default None]`
              Add a suffix to the title.
        """
        import matplotlib.pyplot as plt
        if band_no == 'all':
            for i in range(self.dbands.shape[2]):
                plt.imshow(self.dbands[:,:,i], interpolation='none')
                if suffix == None:
                    plt.title('SG band {0}'.format(i))
                else:
                    plt.title('SG band {0} - {1}'.format(i, suffix))
                plt.show()
                plt.close()
        else:
            plt.imshow(self.dbands[:,:,band_no], interpolation='none')
            if suffix == None:
                plt.title('SG band {0}'.format(band_no))
            else:
                plt.title('SG band {0} - {1}'.format(band_no, suffix))
            plt.show()
            plt.close()


class Whiten(object):
    """Whiten the cube."""

    def __init__(self):
        self.dM = None

    @ApplyInputValidation('Whiten')
    def apply(self, M):
        """
        Whitens a HSI cube. Use the noise covariance matrix to decorrelate
        and rescale the noise in the data (noise whitening).
        Results in transformed data in which the noise has unit variance
        and no band-to-band correlations.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

        Returns: `numpy array`
              A whitened HSI cube (m x n x p).
        """
        h, w, numBands = M.shape
        M = np.reshape(M, (w*h, numBands))
        dM = dnoise.whiten(M)
        self.dM = np.reshape(dM, (h, w, numBands))
        return self.dM

    def get(self):
        """
        Returns: `numpy array`
            The whitened HSI cube (m x n x p).
        """
        return self.dM


class MNF(object):
    """Transform a HSI cube."""

    def __init__(self):
        self.mnf = None
        self.transform = None
        self.wdata = None # temp

    @ApplyInputValidation('MNF')
    def apply(self, M):
        """
        A linear transformation that consists of a noise whitening step
        and one PCA rotation.

        This process is designed to
            * determine the inherent dimensionality of image data,
            * segregate noise in the data,
            * allow efficient elimination and/or reduction of noise, and
            * reduce the computational requirements for subsequent processing.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

        Returns: `numpy array`
              A MNF transformed cube (m x n x p).

        References:
          C-I Change and Q Du, "Interference and Noise-Adjusted Principal
          Components Analysis," IEEE TGRS, Vol 36, No 5, September 1999.
        """
        from sklearn.decomposition import PCA
        w = Whiten()
        wdata = w.apply(M)
        self.wdata = wdata #temp
        h, w, numBands = wdata.shape
        X = np.reshape(wdata, (w*h, numBands))
        self.transform = PCA()
        mnf = self.transform.fit_transform(X)
        self.mnf = np.reshape(mnf, (h, w, numBands))
        return self.mnf

    @XInputValidation('MNF')
    def inverse_transform(self, X):
        """
        Inverse the PCA rotation step. The cube stay
        whitened. Usefull if you want to denoise noisy
        bands before the rotation.

        X: `numpy array`
            A transformed (MNF) cube (m x n x p).

        Return: `numpy array`
            A inverted cube (m x n x p).
        """
        h, w, numBands = X.shape
        X = np.reshape(X, (w*h, numBands))
        M = self.transform.inverse_transform(X)
        M = np.reshape(M, (h, w, numBands))
        return M

    def get_components(self, n):
        """
        Return: `numpy array`
            Return the first n bands (maximum variance bands).
        """
        return self.mnf[:,:,:n]

    def plot_components(self, path, n_first=None, colorMap='jet', suffix=None):
        """
        Plot some bands.

        Parameters:
            path: `string`
              The path where to put the plot.

            n_first: `int [default None]`
                Print the first n components.

            colorMap: `string [default jet]`
              A matplotlib color map.

            suffix: `string [default None]`
              Suffix to add to the title.
        """
        import matplotlib.pyplot as plt
        if n_first != None:
            n = min(n_first, self.mnf.shape[2])
        else:
            n = self.mnf.shape[2]
        plt.ioff()
        for i in range(n):
            plt.imshow(self.mnf[:,:,i], interpolation='none', cmap=colorMap)
            if suffix == None:
                fout = osp.join(path, 'MNF_bandno_{0}.png'.format(i+1))
            else:
                fout = osp.join(path, 'MNF_bandno_{0}_{1}.png'.format(i+1, suffix))
            plt.savefig(fout)
            plt.clf()
        plt.close()

    def display_components(self, n_first=None, colorMap='jet', suffix=None):
        """
        Display some bands.

        Parameters:
            n_first: `int [default None]`
                Display the first n components.

            colorMap: `string [default jet]`
              A matplotlib color map.

            suffix: `string [default None]`
              Suffix to add to the title.
        """
        import matplotlib.pyplot as plt
        if n_first != None:
            n = min(n_first, self.mnf.shape[2])
        else:
            n = self.mnf.shape[2]
        for i in range(n):
            plt.imshow(self.mnf[:,:,i], interpolation='none', cmap=colorMap)
            if suffix == None:
                plt.title('MNF bandno {0}'.format(i+1))
            else:
                plt.title('MNF bandno {0} - {1}.png'.format(i+1, suffix))
            plt.show()
            plt.clf()
        plt.close()
