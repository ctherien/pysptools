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
# speclib.py - This file is part of the PySptools package.
#

"""
SpecLib, USGS06SpecLib classes
"""

from __future__ import print_function

import re
import numpy as np

try:
    import spectral.io.envi as envi
except ImportError:
    pass


class JSONReader(object):
    """
    Open a JSON version of the USGS library and
    return the data and the header.

    The JSON version is made of two files:
        * s06av95a_envi.jdata
        * s06av95a_envi.jhead

    You create the reader this way (on Windows):
        rd = JSONReader(r'c:\\\somepath\\\s06av95a_envi')

    And pass it to the SpecLib class this way:
        lib = SpecLib(rd)
    """

    def __init__(self, fname):
        import json
        import os.path as osp
        data_file = osp.join(fname+'.jdata')
        with open(data_file, 'r') as content_file:
            self.spectra = np.array(json.loads(content_file.read()))
        info_file = osp.join(fname+'.jhead')
        with open(info_file, 'r') as content_file:
            self.head = json.loads(content_file.read())
        self.wvl = self.head['wavelength']
        self.spectra_names = self.head['spectra names']


class EnviReader(object):
    """
    Open a ENVI version of the USGS library and
    return the data and the header.

    The ENVI version is made of two files:
        * s06av95a_envi.hdr
        * s06av95a_envi.sli

    You create the reader this way (on Windows):
        rd = EnviReader(r'c:\\\somepath\\\s06av95a_envi.hdr')

    And pass it to the SpecLib class this way:
        lib = SpecLib(rd)
    """

    def __init__(self, fname):
        self.f_handle = envi.open(fname)
        self.spectra = self.f_handle.spectra
        self.head = envi.read_envi_header(fname)
        self.wvl = [float(x) for x in self.head['wavelength']]
        self.spectra_names = self.head['spectra names']


class SpecLib(object):

    def __init__(self, reader):
        self.spectra = reader.spectra
        self.wvl = reader.wvl
        self.spectra_names = reader.spectra_names

    def get_wvl(self):
        return self.wvl

    def get_dim(self):
        return len(self.spectra_names)


# to return a list or numpy array ?
class USGS06SpecLib(SpecLib):
    """Load the library and add get and search functionnality.

        Parameters:
            reader: EnviReader or JSONReader instance
    """

    def __init__(self, reader):
        SpecLib.__init__(self, reader)

    def _norm_name(self, str):
        s = str.replace('"','')
        s = s[0].upper()+s[1:].lower()
        s = s.rstrip(' ')
        s = s.replace(' ','_')
        s = s.replace('-','_')
        s = s.replace('/','_')
        s = s.replace('+','_')
        s = s.replace('(','_')
        s = s.replace(')','')
        return s

    def _norm_sample_id(self, str):
        s = str
        s = s.replace('/','_')
        return s

    def _regex_usgs_spec_name(self, sn):
        p = re.compile(r'([A-Za-z0-9\_\-\/\+]+)\s*([A-Za-z0-9\_\-\/\+\.]+)(.*)$')
        m = p.match(sn)
        mineral = self._norm_name(m.group(1))
        sample_id = self._norm_sample_id(m.group(2))
        description = m.group(3)
        return mineral, sample_id, description

    def get(self, idx):
        """
        Return the spectrum at index idx.
        Indexing start at zero.

        Parameters:
            idx: `int`
                The index of the spectrum to fetch.

        Returns: `list`
            Spectrum (p).
        """
        return self.spectra[idx].tolist()

    def get_next(self):
        """
        Iterator, scan the library and return at each step:
        spectrum, mineral, sample_id, description and index.

        Return: `tuple`
            Spectrum, mineral, sample_id, description and index.
        """
        for i, spec_name in enumerate(self.spectra_names):
                y = self.spectra[i].tolist()
                if y[0] > 0 and np.min(y) > 0:
                    mineral, sample_id, descrip = self._regex_usgs_spec_name(spec_name)
                    yield y, mineral, sample_id, descrip, i

    def get_substance(self, substance, sample=None):
        """
        Iterator, scan the library, verify the conditions 'substance'
        and/or 'sample' and if true return:
        spectrum, sample_id, description and index.

        Parameters:
            substance: `string`
                A substance name.

            sample: `string [default None]`
                A sample.

        Returns: `tuple`
            Spectrum, mineral, sample_id, description and index.
        """
        for i, spec_name in enumerate(self.spectra_names):
                y = self.spectra[i].tolist()
                if y[0] > 0 and np.min(y) > 0:
                    mineral, sample_id, descrip = self._regex_usgs_spec_name(spec_name)
                    if sample != None:
                        if mineral == substance and sample_id == sample:
                            yield y, sample_id, descrip, i
                    else:
                        if mineral == substance:
                            yield y, sample_id, descrip, i

    def distance_match(self, s, distfn='SAM'):
        """
        Scan the library and return the index of the spectrum that
        have the smallest distance to 's'.

        Parameters:
            s: `list`
                A spectrum.

            distfn: `function [default 'SAM']`
                One of the function: SAM, SID, chebyshev, NormXCorr.

        Returns: `tuple`
            Spectrum, index.
        """
        import pysptools.distance as dst
        if distfn == 'SAM': fn = dst.SAM
        elif distfn == 'SID': fn = dst.SID
        elif distfn == 'chebyshev': fn = dst.chebyshev
        elif distfn == 'NormXCorr': fn = dst.NormXCorr
        else:
            print('USGS06SpecLib.distance_match needs a valid distance function')
            return None
        scores = []
        index = []
        for spectrum, mineral, sample_id, descrip, idx in self.get_next():
            scores.append(fn(np.array(spectrum), np.array(s)))
            index.append(idx)
        if distfn == 'NormXCorr':
            where = np.argmax(scores)
        else:
            where = np.argmin(scores)
        true_idx = index[where]
        match_spectrum = self.spectra[true_idx].tolist()
        return match_spectrum, true_idx
