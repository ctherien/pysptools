"""
Plot the convex hull and the features for 4 substances.
"""

import os
import sys
import pysptools.spectro as spectro


class SpecLib(object):

    def __init__(self, lib_name):
        rd = spectro.EnviReader(lib_name)
        self.lib = spectro.USGS06SpecLib(rd)

    def get(self, substance, sample):
        for spectrum, sample_id, descrip, idx in self.lib.get_substance(substance, sample):
            return spectrum

    def get_wvl(self):
        return self.lib.get_wvl()


def plot_convex_hull(path_out, lib, substance, sample):
    spectrum = lib.get(substance, sample)
    wvl = lib.get_wvl()

    schq = spectro.SpectrumConvexHullQuotient(spectrum, wvl)
    plot_name = '{0}_{1}'.format(substance, sample)
    schq.plot(path_out, plot_name)


def extract_features(path_out, lib, baseline, substance, sample):
    """
    Process the s06av95a_envi file and extract the <substance> and/or <sample>
    features according to the <baseline> value.
    """
    spectrum = lib.get(substance, sample)
    wvl = lib.get_wvl()
    fea = spectro.FeaturesConvexHullQuotient(spectrum, wvl, baseline=baseline)
    plot_name = '{0}_{1}'.format(substance, sample)
    fea.plot(path_out, plot_name, feature='all')


substances = [('Biotite', 'WS660'),
            ('Chalcedony', 'CU91-6A'),
            ('Kaolinite', 'CM7'),
            ('Gibbsite', 'HS423.3B')]


if __name__ == '__main__':

    data_path = os.environ['PYSPTOOLS_USGS']
    home = os.environ['HOME']
    result_path = os.path.join(home, 'results')
    if os.path.exists(result_path) == False:
        os.makedirs(result_path)

    hdr_name = 's06av95a_envi.hdr'
    lib_path = os.path.join(data_path, hdr_name)

    lib = SpecLib(lib_path)
    base = 0.93
    for substance,sample in substances:
        plot_convex_hull(result_path, lib, substance, sample)
        extract_features(result_path, lib, base, substance, sample)
