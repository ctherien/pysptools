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
# test_hull.py - This file is part of the PySptools package.
#

"""
The following classes are tested:
    SpectrumConvexHullQuotient
    FeaturesConvexHullQuotient
"""


# What this program do:
# Create a plot of the spectrum and the hull quotient for each
# entry of the usgs spectral library

# How to use it:
# First you need to download the following files from the usgs ftp site:
#  s06av95a_envi.hdr and s06av95a_envi.sli (ENVI file format)
#
# The spectral library home page is : http://speclab.cr.usgs.gov/spectral-lib.html
# Go to the ftp site:
#  ftp://ftpext.cr.usgs.gov/pub/cr/co/denver/speclab/pub/spectral.library/splib06.library/Convolved.libraries/
#
# and pick s06av95a_envi.hdr and s06av95a_envi.sli
#
# Create a "usgs" directory under pysptools directory and move the files to
# them or edit the data_path below
#
# Run this program, the result is a plot for each spectrum and is dropped in
# the "results/qhull" directory
#

from __future__ import print_function

import os
import os.path as osp
import pysptools.spectro as spectro


def extract_features(fname, file_type, path_out, baseline, substance, sample=None):
    """
    Process the s06av95a_envi file and extract the <substance> and/or <sample>
    features according to the <baseline> value.
    """
    print('Running extract_features')
    if file_type == 'ENVI': rd = spectro.EnviReader(fname)
    if file_type == 'JSON': rd = spectro.JSONReader(fname)
    lib = spectro.USGS06SpecLib(rd)
    wvl = lib.get_wvl()
    for spectrum, sample_id, descrip, idx in lib.get_substance(substance, sample):
        fea = spectro.FeaturesConvexHullQuotient(spectrum, wvl, baseline=baseline)
        plot_name = '{0}_{1}'.format(substance, sample_id)
        fea.plot_convex_hull_quotient(path_out, plot_name)
        fea.plot(path_out, plot_name, feature='all')
        #fea.plot(path_out, plot_name, feature=10)
        fea.print_stats('all')


def batch_usgs_spec_plot(fname, file_type, path_out):
    """
    Process the s06av95a_envi file and plot for each spectrum the spectrum,
    the convex hull and the convex hull quotient.
    """
    print('Running batch_usgs_spec_plot')
    if file_type == 'ENVI': rd = spectro.EnviReader(fname)
    if file_type == 'JSON': rd = spectro.JSONReader(fname)
    lib = spectro.USGS06SpecLib(rd)
    wvl = lib.get_wvl()

    for spectrum, mineral, sample_id, descrip, idx in lib.get_next():
        schq = spectro.SpectrumConvexHullQuotient(spectrum, wvl)
        plot_name = '{0}_{1}_{2}'.format(idx, mineral, sample_id)
        schq.plot(path_out, plot_name)


def tests():
    usgs_path = os.environ['PYSPTOOLS_USGS']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results/qhull')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    hdr_name = 's06av95a_envi.hdr'
    fname = os.path.join(usgs_path, hdr_name)

    # Take the USGS USGS06 spectral library and translate to a
    # quotient hull all the spectra. Print it.
    batch_usgs_spec_plot(fname, 'ENVI', result_path)

    # Take the USGS USGS06 spectral library and search for the biotite spectrum.
    # When found, extract and print the features. The baseline is set to 0.93.
    result_path = os.path.join(home, 'results/features')
    if os.path.exists(result_path) == False:
        os.makedirs(result_path)

    extract_features(fname, 'ENVI', result_path, 0.93, 'Biotite')
    extract_features(fname, 'ENVI', result_path, 0.93, 'Biotite', 'WS660')


if __name__ == '__main__':
    tests()
