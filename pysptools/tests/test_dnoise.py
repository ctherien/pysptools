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
# test_dnoise.py - This file is part of the PySptools package.
#

"""
The following functions are tested:
    SavitzkyGolay
    Whiten
    MNF
"""

from __future__ import print_function

import os
import os.path as osp
import pysptools.util as util
import pysptools.noise as ns
import pysptools.skl as skl


def SavitzkyGolay_spectra_filter(data, path):
    sg = ns.SavitzkyGolay()
#    fdata = sg.denoise_spectra(data, 11, 4, deriv=2)
    fdata = sg.denoise_spectra(data, 11, 4, deriv=0)
    sg.plot_spectrum_sample(data, path, 0, 1)
    return fdata


def SavitzkyGolay_bands_filter(data, path):
    sg = ns.SavitzkyGolay()
    fdata = sg.denoise_bands(data, 5, 2)
    sg.plot_bands_sample(path, 5)
    return fdata


def whiten(data):
    w = ns.Whiten()
    return w.apply(data)


def MNF(data, n_components, path):
    mnf = ns.MNF()
    mnf.apply(data)
    mnf.plot_components(path, 3)
    # get the first n_components
    return mnf.get_components(n_components)


def MNF_reduce_component_2_noise_and_invert(data):
    # Reduce the second component noise and
    # return the inverse transform
    mnf = ns.MNF()
    tdata = mnf.apply(data)
    dn = ns.SavitzkyGolay()
    tdata[:,:,1:2] = dn.denoise_bands(tdata[:,:,1:2], 15, 2)
    # inverse_transform remove the PCA rotation,
    # we obtain a whitened cube with
    # a noise reduction for the second component
    return mnf.inverse_transform(tdata)


def test_whiten(n_clusters, data, result_path):
    print('Testing whiten')
    wdata = whiten(data)
    km = skl.KMeans()
    km.predict(wdata, n_clusters)
    km.plot(result_path, colorMap='jet', suffix='whiten')

def test_MNF(n_clusters, n_components, data, result_path):
    print('Testing MNF')
    tdata = MNF(data, n_components, result_path)
    km = skl.KMeans()
    km.predict(tdata, n_clusters)
    km.plot(result_path, colorMap='jet', suffix='MNF')

    print('Testing MNF with component 2 noise reduction')
    idata = MNF_reduce_component_2_noise_and_invert(data)
    km = skl.KMeans()
    km.predict(idata, n_clusters)
    km.plot(result_path, colorMap='jet', suffix='MNF_with_component_2_noise_reduction')


def test_SavitzkyGolay(n_clusters, data, result_path):
    print('Testing SavitzkyGolay bands filter')

    tdata = SavitzkyGolay_bands_filter(data, result_path)
    km = skl.KMeans()
    km.predict(tdata, n_clusters)
    km.plot(result_path, colorMap='jet', suffix='SG_bands_filter')

    tdata = SavitzkyGolay_spectra_filter(data, result_path)
    km = skl.KMeans()
    km.predict(tdata, n_clusters)
    km.plot(result_path, colorMap='jet', suffix='SG_spectra_filter')


def tests():
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    sample = '92AV3C.hdr'

    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)

    n_clusters = 5
    km = skl.KMeans()
    km.predict(data, n_clusters)
    km.plot(result_path, colorMap='jet', suffix='data')

    n_components = 40
    test_MNF(n_clusters, n_components, data, result_path)
    test_whiten(n_clusters, data, result_path)
    test_SavitzkyGolay(n_clusters, data, result_path)


if __name__ == '__main__':
    import sys
    print(sys.version_info)
    tests()
