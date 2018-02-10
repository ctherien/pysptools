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
# test_cls.py - This file is part of the PySptools package.
#


"""
Python Spectral Tools

The following functions and classes are tested:
    SAM
    SID
    NormXCorr
    NFINDR
    FIPPI
    ATGP
"""

from __future__ import print_function

import os
import os.path as osp
import cProfile, pstats

import pysptools.classification as cls
import pysptools.util as util
import pysptools.eea as eea


_doProfile = False
def profile():
    if _doProfile == True:
        pr = cProfile.Profile()
        pr.enable()
        return pr

def stat(pr):
    if _doProfile == True:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats('time')
        ps.print_stats()


def parse_ENVI_header(fname, head):
    # Parse a ENVI header and fill up
    # the axes dictionary. Later
    # is needed by the PPI ... plot method.
    axes = {}
    if fname == '92AV3C.hdr' or fname == '92AV3C':
        axes['wavelength'] = head['wavelength']
        axes['x'] = 'Wavelength (Unknown)'
        axes['y'] = 'Reflectance'
    return axes


def plot(image, colormap, desc, path):
    import matplotlib.pyplot as plt
    plt.ioff()
    img = plt.imshow(image, interpolation='none')
    img.set_cmap(colormap)
    plt.colorbar()
    fout = osp.join(path, 'plot_{0}.png'.format(desc))
    plt.savefig(fout)
    plt.clf()


def NFINDR(data, wvl, path):
    print('Running NFINDR')
    nfindr = eea.NFINDR()
    U = nfindr.extract(data, 5, maxit=5, normalize=False, ATGP_init=True)
    nfindr.plot(path, axes=wvl, suffix='test_cls')
    # U[0,:] is a false positive, we remove it
    return U[[1,2,3,4],:]


def ROI(data, path):
    r = util.ROIs(data.shape[0], data.shape[1])
    r.add('Area', {'rec': (30,30,100,100)})
    r.plot(path)
    return r


def test_SID_single(data, E, path):
    print('Testing SID - plot_single_map -')
    mp = cls.SID()
    cmap = mp.classify(data, E, threshold=0.2)
    m1 = mp.get_SID_map()
    m2 = mp.get_single_map(1)
    plot(m2, 'jet', 'SID_single_1', path)
    mp.plot_single_map(path, 'all', constrained=False, colorMap='gist_earth', stretch=True, suffix='ucon_single')
    mp.plot_single_map(path, 'all', constrained=True, suffix='con_single')


def test_SAM_single(data, E, path):
    print('Testing SAM - plot_single_map -')
    mp = cls.SAM()
    cmap = mp.classify(data, E, threshold=0.3)
    m1 = mp.get_angles_map()
    s = mp.get_angles_stats()
#    c = mp.get_single_map(1, constrained='allo')
    m2 = mp.get_single_map(1)
    plot(m2, 'jet', 'SAM_single_1', path)
    mp.plot_single_map(path, 'all', constrained=False, colorMap='gist_earth', stretch=False, suffix='ucon_single')
    #mp.plot_single_map(path, 'all', constrained=1, colorMap='gist_earth', stretch=True, suffix=[1,2])
    mp.plot_single_map(path, 'all', constrained=True, suffix='con_single')


def test_NormXCorr_single(data, E, path):
    print('Testing NormXCorr - plot_single_map -')
    mp = cls.NormXCorr()
    cmap = mp.classify(data, E, threshold=0.1)
    m = mp.get_single_map(1)
    mp.plot_single_map(path, 'all', constrained=False, colorMap='gist_earth', stretch=True, suffix='ucon_single')
    mp.plot_single_map(path, 'all', constrained=True, suffix='con_single')


lbl = ['EM1','EM2','EM3','EM4']


def test_SID(data, E, roi, path):
    print('Testing SID - plot -')
    mp = cls.SID()
    pr = profile()
    mp.classify(data, E, mask=roi.get_mask())
    stat(pr)
    print(str(mp))
    mp.plot(path, colorMap='Paired', suffix='t1')
    mp.plot(path, labels=lbl, colorMap='Paired', suffix='labels_t1')
    mp.plot_histo(path)
    cmap = mp.classify(data, E, threshold=0.03)
    mp.plot(path, interpolation=None, colorMap='Paired', suffix='t2')
    mp.plot_histo(path)
    cmap = mp.classify(data, E, threshold=[0.1,0.1,0.05,0.1])
    mp.plot(path, mask=roi.get_mask(), colorMap='Paired', suffix='t3')
    mp = cls.SID()
    cmap = mp.classify(data, E[2,:])
    mp.plot(path, colorMap='gist_earth', suffix='t4')


def test_SAM(data, E, roi, path):
    print('Testing SAM - plot -')
    mp = cls.SAM()
    pr = profile()
    mp.classify(data, E, mask=roi.get_mask())
#    mp.classify(data, E)
    stat(pr)
    print(str(mp))
    m = mp.get_angles_stats()
    m = mp.get_angles_map()
    mp.plot(path, colorMap='Paired', suffix='t1')
    mp.plot(path, labels=lbl, colorMap='Paired', suffix='labels_t1')
    mp.plot_histo(path)
    cmap = mp.classify(data, E, threshold=0.05)
    #cmap = mp.classify(np.zeros((2,2)), E, threshold=0.05)
    #cmap = mp.classify('allo', E, threshold=0.05)
    #cmap = mp.classify(data, E, threshold=0.05)
    mp.plot(path, interpolation=None, colorMap='Paired', suffix='t2')
    #mp.plot(path, labels=1.0, interpolation=None, colorMap='Paired', suffix='t2')
    #mp.plot(path, mask=np.zeros((2,2,2)), interpolation=None, colorMap='Paired', suffix='t2')
    mp.plot_histo(path)
    cmap = mp.classify(data, E, threshold=[0.1,0.1,0.05,0.1])
    mp.plot(path, mask=roi.get_mask(), colorMap='Paired', suffix='t3')
    mp = cls.SAM()
    cmap = mp.classify(data, E[2,:])
    mp.plot(path, colorMap='gist_earth', suffix='t4')


def test_NormXCorr(data, E, roi, path):
    print('Testing NormXCorr - plot -')
    mp = cls.NormXCorr()
    pr = profile()
    mp.classify(data, E, mask=roi.get_mask())
#    mp.classify(data, E)
    stat(pr)
    print(str(mp))
    mp.plot(path, colorMap='Paired', suffix='t1')
    mp.plot(path, labels=lbl, colorMap='Paired', suffix='labels_t1')
    mp.plot_histo(path)
    cmap = mp.classify(data, E, threshold=0.05)
    mp.plot(path, interpolation=None, colorMap='Paired', suffix='t2')
    cmap = mp.classify(data, E, threshold=[0.1,0.1,0.05,0.1])
    mp.plot(path, mask=roi.get_mask(), colorMap='Paired', suffix='t3')
    mp = cls.NormXCorr()
    cmap = mp.classify(data, E[2,:])
    mp.plot(path, colorMap='gist_earth', suffix='t4')


def ATGP(data, wvl, mask, path):
    print('Testing ATGP')
    atgp = eea.ATGP()
    U = atgp.extract(data, 4, normalize=True)
    atgp.plot(path, suffix='test2')
    return U


def NNLS(data, U, umix_source, mask, path):
    import pysptools.abundance_maps as amp
    print('  Testing NNLS')
    nnls = amp.NNLS()
    amaps = nnls.map(data, U, normalize=True)
    nnls.plot(path, colorMap='jet', suffix=umix_source)
    nnls.plot(path, interpolation='spline36', suffix=umix_source+'_spline36')
    return amaps


def test_AbundanceClassification(data, path):
    U = ATGP(data, None, None, path)
    mps = NNLS(data, U, 'ATGP', None, path)
    amapcls = cls.AbundanceClassification()
#    thecmap = amapcls.classify(mps[:,:,[1]], threshold=0.4)
    thecmap = amapcls.classify(mps[:,:,[0,1,2]], threshold=[0.4,0.6,0.2])
    print(str(amapcls))
#    thecmap = amapcls.classify(mps)
    amapcls.plot(path, colorMap='Paired')
#    amapcls.plot(result_path, colorMap='gist_earth')


def test_one_spectrum(data, E, roi, path):
    # classify one pixel
    print('Testing SAM one spectrum')
    for i in range(E.shape[0]):
        cl = cls.SAM()
        cl.classify(data, E[i], mask=roi.get_mask())
        #cl.classify(data, E[i])
        cl.plot(path, colorMap='gist_earth', suffix='t5_{0}'.format(i+1))
    print('Testing SID one spectrum')
    for i in range(E.shape[0]):
        cl = cls.SID()
        cl.classify(data, E[i], mask=roi.get_mask())
        #cl.classify(data, E[i])
        cl.plot(path, colorMap='gist_earth', suffix='t5_{0}'.format(i+1))
    print('Testing NormXCorr one spectrum')
    for i in range(E.shape[0]):
        cl = cls.NormXCorr()
        cl.classify(data, E[i], mask=roi.get_mask())
        #cl.classify(data, E[i])
        cl.plot(path, colorMap='gist_earth', suffix='t5_{0}'.format(i+1))


def tests():
    import pysptools.util as util
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = osp.join(home, 'results')
    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    sample = '92AV3C.hdr'

    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)
    axes = parse_ENVI_header(sample, header)

    U = NFINDR(data, axes, result_path)
    r = ROI(data, result_path)

    test_SID(data, U, r, result_path)
    test_SAM(data, U, r, result_path)
    test_NormXCorr(data, U, r, result_path)
    test_SID_single(data, U, result_path)
    test_SAM_single(data, U, result_path)
    test_NormXCorr_single(data, U, result_path)
    test_AbundanceClassification(data, result_path)
    test_one_spectrum(data, U, r, result_path)


if __name__ == '__main__':
    import sys
    print(sys.version_info)
    tests()
