Convex hull and features extraction
+++++++++++++++++++++++++++++++++++

This is a quick overview of the convex hull removal and features extraction functions. They are part of the spectro module.

The overall goal is to extract the spectrum features. To identify a spectral feature by its wavelength position and shape, it must be isolated from effects like level changes and slopes. The first step is to normalize the spectrum by applying it a continuum removal algorithm. There is two ways of doing it: by division in reflectance, transmittance, and emittance spectra or by subtraction with absorbance or absorption coefficient spectra. The former is what is implemented by the pysptools library (see *SpectrumConvexHullQuotient* class).

The script listed below open the USGS 2006 library and read four spectra. They are:

	* Biotite WS660
	* Chalcedony CU91-6A
	* Kaolinite CM7
	* Gibbsite HS423.3B

Next, the script compute the convex hull removal algorithm on these spectra. For the results, see figure 1.

.. figure:: ./chull/chull_ch.png
   :scale: 100 %
   :align: center
   :alt: None

   Figure 1: continuum removed spectra.

When the convex hull is removed, the absorption features can be isolated and identified. In fact, a call to the init class method *FeaturesConvexHullQuotient()* do all the job in one call. The next figures shows, for each spectrum, the features extracted. The number of features extracted can be controled by ajusting the baseline parameter. This parameter may be different spectrum by spectrum. The class *FeaturesConvexHullQuotient* have a method to output the features statistics.

.. figure:: ./chull/chull_gibbsite_features.png
   :scale: 100 %
   :align: center
   :alt: None

   Figure 2: features for the gibbsite sample.

.. figure:: ./chull/chull_kaolinite_features.png
   :scale: 100 %
   :align: center
   :alt: None

   Figure 3: features for the kaolinite sample.


The code follow::

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
