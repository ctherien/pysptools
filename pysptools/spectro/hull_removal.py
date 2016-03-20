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
# hull_removal.py - This file is part of the PySptools package.
#

"""
convex_hull_removal function
"""

from __future__ import division

from scipy import interpolate
from . import _jarvis


def convex_hull_removal(pixel, wvl):
    """
    Remove the convex-hull of the signal by hull quotient.

    Parameters:
        pixel: `list`
            1D HSI data (p), a pixel.
        wvl: `list`
            Wavelength of each band (p x 1).

    Results: `list`
        Data with convex hull removed (p).

    Reference:
        Clark, R.N. and T.L. Roush (1984) Reflectance Spectroscopy: Quantitative
        Analysis Techniques for Remote Sensing Applications, J. Geophys. Res., 89,
        6329-6340.
    """
    points = list(zip(wvl, pixel))
    # close the polygone
    poly = [(points[0][0],0)]+points+[(points[-1][0],0)]
    hull = _jarvis.convex_hull(poly)
    # the last two points are on the x axis, remove it
    hull = hull[:-2]
    x_hull = [u for u,v in hull]
    y_hull = [v for u,v in hull]

    tck = interpolate.splrep(x_hull, y_hull, k=1)
    iy_hull = interpolate.splev(wvl, tck, der=0)

    norm = []
    for ysig, yhull in zip(pixel, iy_hull):
        if yhull != 0:
            norm.append(ysig/yhull)
        else:
            norm.append(1)

    return norm, x_hull, y_hull
