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
# pwc.py - This file is part of the PySptools package.
#

"""
bilateral function
"""

from __future__ import print_function

import numpy as np


def bilateral(y, soft, beta, width, display=1, stoptol=1**-3, maxiter=50):
    """
    Performs PWC denoising of the input signal using hard or soft kernel
    bilateral filtering.

    Parameters:
        y: `numpy array`
            Original signal to denoise of length N.

        soft: `int`
            Set this to 1 to use the soft Gaussian kernel, else uses
            the hard kernel.

        beta: `int`
             Kernel parameter. If soft Gaussian kernel, then this is the
             precision parameter. If hard kernel, this is the kernel
             support.

        width: `int`
             Spatial kernel width W.

        display: `int [default 1]`
             Set to 0 to turn off progress display, 1 to turn
             on. If not specifed, defaults to progress display on.

        stoptol: `float [default 1**-3]`
             Precision of estimate as determined by square
             magnitude of the change in the solution. If not specified,
             defaults to 1e-3.

        maxiter: `int [default 50]`
             Maximum number of iterations. If not specified,
             defaults to 50.

    Results: `numpy array`
             Denoised output signal.

    Reference:
        (c) Max Little, 2011. If you use this code for your research, please cite:
        M.A. Little, Nick S. Jones (2011)
        "Generalized Methods and Solvers for Noise Removal from Piecewise
        Constant Signals: Part I and II"
        Proceedings of the Royal Society A (in press).

        See http://www.maxlittle.net/

    """

    def gen_ones(x, width):
        if x <= width:
            return 1
        else:
            return 0

    if y.ndim != 1:
        error = "bilateral only accepts 1 dimension arrays"
        raise util.SpToolsException(error)

    N = y.shape[0]

    # Construct bilateral sequence kernel
    w = np.zeros((N,N))
    i = np.ones(N)
    j = np.arange(N)

    for k in range(N):
        w[k,:] = abs(k*i - j) <= width

    xold = y           # Initial guess using input signal
    d = np.zeros((N,N))

    if display == 1:
        if soft == 1:
            print('Soft kernel\n')
        else:
            print('Hard kernel\n')
        print('Kernel parameters beta={0}, W={1}\n'.format(beta, width))
        print('Iter# Change')

    iter = 1
    gap = float('inf')
    while (iter < maxiter):

        if display == 1:
            print('{0} {1}'.format(iter, gap))

        # Compute pairwise distances between all samples
        for i in range(N):
            d[:,i] = 0.5*(xold-xold[i])**2

        # Compute kernels
        if soft == 1:
            W = np.exp(np.dot(-beta, d)) * w # Gaussian (soft) kernel
        else:
            W = (d <= beta**2)*w   # Characteristic (hard) kernel

        # Do kernel weighted mean shift update step
        # attention, pas sur, pas sur
        xnew = np.dot(W.T, xold) / np.sum(W, axis=1) # pas sur

        gap = np.sum((xold - xnew)**2)

        # Check for convergence
        if gap < stoptol:
            if display == 1:
                print('Converged in %d iterations\n' % iter);
            break;

        xold = xnew
        iter = iter + 1

    if display == 1:
        if iter == maxiter:
            print('Maximum iterations exceeded\n')

    return xnew

