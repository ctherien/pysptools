Release notes
*************

Version 0.13.5 (beta)
=====================

It's mainly a maintenance version with a few fixes. Following is a list of the new features.

* A __str__ method is added to the abundance_maps, classification, detection and eea modules classes.

* A new parameter 'mask' is added to the classify method of the classification module classes. And the same
  to the extract method of the eea module classes. For NFINDR and ATGP the code is reorganized accordingly.
  The mask is a binary one. Only the selected pixels by the mask are used. The motivation is to
  improve performance when we work in a region of the hyperspectral cube.

* To the methods plot_components and display_components of the MNF class, a colorMap parameter is added and the interpolation is now set to 'none' by default.

* The cython version for the NFINDR algorithm is removed from the pysptools development path.
  It is now in is own project. Follow the link `eealgo <https://github.com/ctherien/eealgo/>`_

* Clean up of the DATA file. The DATA fike contains the data to run the tests suite. The new version is put on the sourceforge site. Follow the link `data file <http://sourceforge.net/projects/pysptools/files/>`_

Version 0.13.3 (beta) and version 0.13.4 (beta)
===============================================

The library is now compatible with Python 2.7 and 3.x. Otherwise, a few fixes to mute some warnings. In details:

* Compatibility for Python 2.7 and 3.x was improved. The same archive run on both.

* A new parameter 'columns' was added to the plot and display methods for the abundances_map module
  classes. When you use this parameter, all the abundance maps are rendered in one file for the plot
  method and in one image for the display method. 'columns' control the number of columns.

* A new parameter 'mask' was added to the map method for the abundances_map module classes.
  The mask is a binary one. Only the selected pixels by the mask are unmixed. The motivation is to
  improve performance when we work in a region of the hyperspectral cube.

Version 0.13.2 (beta)
=====================

This is a maintenance version with a few new features. This version will impact your code.

* Some classes and functions are moved to the *util* module:

 * ROIs.
 * the *formating* module content: convert2d, convert3d, normalize.
 
* The module *formating* is removed.

* New: the input validation is completely reworked. The validation is now composed of a main class *InputValidation* that live in the *util* module and of secondary functions living in the differents inval.py files presents in some modules where the inputs are validated. These functions support the decorator pattern.

* New: input validating decorator functions are added to many of the classes composing the library.
 
* New: the class *AbundanceClassification* is added to the module *classification*; this class use an abundance maps array as input and output a classification map.

* Fix: nfindr.pyx and the related files to compile the NFINDR cython version are with the distribution.

* Documentation fix: for the ROIs class add method, the documentation string was:

            rois: `dictionary list`
              Each parameter, a dictionary, represent a rectangle or a polygon.
              For a rectangle: {'rec': (upper_left_x, upper_left_y, lower_right_x, lower_right_y)}
              For a polygone: {'poly': ((x1,y1),(x2,y2), ...)}, the polygon don't need to be close.
              You can define one or more rectangle and/or polygon for a same cluster.
              The polygon and the rectangle must be VALID.

  But the meaning is:

            rois: `dictionary list`
              Each parameter, a dictionary, represent a rectangle or a polygon. They use matrix coordinates.
              For a rectangle: {'rec': (upper_left_line, upper_left_column, lower_right_line, lower_right_column)}
              For a polygone: {'poly': ((l1,c1),(l2,c2), ...)}, **l** stand for line and **c** for column. The polygon don't need to be close.
              You can define one or more rectangle and/or polygon for a same cluster.
              The polygon and the rectangle must be well formed.
  
Version 0.13.1 (beta)
=====================

This is a maintenance version, many fixes and improvements. All the display and plot methods was revised. This version may have an impact on your code.

* The dependency between the ENVI header file and the NFINDR, PPI, ATGP and FIPPI plot and display methods is removed. You should use this format for the axes dictionary:

 * axes['wavelength'] : a wavelengths list (1D python list). If None or not specified the list is automaticaly numbered starting at 1.
 * axes['x'] : the x axis label, 'Wavelength' if None or not specified. axes['x'] is copied verbatim.
 * axes['y'] : the y axis label, 'Brightness' if None or not specified. axes['y'] is copied verbatim.

* Fix: classes NFINDR, ATGP: internaly the mask have a wrong format resulting in ineffective masking.

* Fix: classes NFINDR, PPI, ATGP and FIPPI: when calling the method get_idx(), the output coordinates have their axis inverted, e.g.: output is [(y1,x1),(y2,x2)...] and suppose to be [(x1,y1),(x2,y2)...].

* Fix: functions mATGP and ATGP (Note: the classes NFINDR and ATGP call these functions): the last endmember returned by ATGP is always false (not a endmember), this problem is caused by an internal indexing error. The cython version is fixed accordingly.

* Fix: util.display_linear_stretch(): xrange replaced by range to keep Python 3 compatibility.

* Fix: spectro.SpectrumConvexHullQuotient(): normalize parameter was ineffective, fixed.

* Fix: spectro.FeaturesConvexHullQuotient(): the normalize parameter was added to the __init__() method.

* Major code refactoring of the class classification.Output.

* Added: a mask and a labels parameters was added to the class classification.Output.

* Added: a mask and an interpolation parameters to the plot() and display() methods for the Classes UCLS, NNLS, FCLS.

* Added: stretch and colorMap parameters to the methods plot_single_map and display_single_map for the classes SAM, SID and NormXCorr.

* Added: labels, mask and interpolation parameters to the plot and display methods for the classes SAM, SID and NormXCorr.

* Class KMeans and class SVC use now the class classification.Output.

* The dependencies between the classes ROIs and SVC are removed. The class ROIs can now plot itself and use at every place a parameter mask is present.

* Updated: all nbex_pine_creek1 to nbex_pine_creek4.

* Updated: example ex_smokestack_v3.py: the call to SAM is replaced by a call to NormXCorr.

* Now the examples run on both Python 2.7 and 3.3 (see ex_methanol_burner_v2.py, ex_convexhull.py, ex_hematite_v2.py and ex_smokestack_v3.py).

* A stretch mode is added to the method classification.(SAM,SID,NormXCorr).plot_single_map()

... and others small fixes.

Version 0.13.0 (beta)
=====================

* The class classification.SVM adapt the scikit-learn implementation of Support Vector Machines to the analysis of hyperspectral cubes.

* An example of using the SVC class with the Pine Creek cube.

version 0.12.2 (beta)
=====================

This is a maintenance version: code review, bugs fix, classes refactoring and many small improvements that have not a direct impact on the functionality.

New
---

* Full Python 3.3 compatibility. However, there is a drawback, as the SPy software ENVI file reader is not ported to Python 3.3, instead, pysptools use a JSON version for the USGS spectral library. See the spectro module documentation.

* The scikit-learn KMeans class is wrapped to make it user-friendly when applied to a HS cube. This new class live in the classification module.

* A new Pine Creek example that illustrate the use of KMeans with unmixing.

Reorganisation
--------------

The module classifiers is renamed classification (Yes, again! Sorry)

version 0.12.1 (beta)
=====================

New
---

* This version adds compatibility to the IPython Notebook. A *display* method is introduced for many classes. This one use matplotlib and have the same role than the *plot* method. Calling *display* show figures embedded in the Notebook.

* The source is ported to Pyhton 3.3. This porting is not integral. Because the SPy library run on Python <= 2.7, the spectro module is not part of the porting.

New examples
------------

This version adds compatibility to the IPython Notebook. Five examples present this feature.

* Methanol gas
* Hematite drill core
* Convex hull
* Pine Creek 1
* Pine Creek 2

Fix
---

* Function distance.SID: the division at the line "p = (s1 / np.sum(s1)) and (s2 / np.sum(s2))" is now always real with the inclusion of a "__future__ division" declaration. That was not the case before, but this is a minor problem as cubes are always made of real.

* Function denoise.whiten: the line "S_1_2 = S**-1/2" is changed to "S_1_2 = S**(-0.5)". This is a bug and have a direct impact on the classes Whiten and MNF.

version 0.12.0 (beta)
=====================

New
---

* A first version of the HySime hyperspectral signal subspace estimation.
* SAM, SID and NormXCorr classifiers classes can now classify one spectrum at a time. Previously, they asked for two or more spectra to be classified.
* A parameter *noise_whitening* was added to the HfcVd class. When set to *True* a whiten is applied before calling HfcVd and this way implement the NWHFC function.

Fix
---

* The variance calculation for HfcVd was wrong - it was integer based but it is assume to be float based - fixed.
* The call to normalize() inside the count() method that belong to the HfcVd class was removed. Now, data conditioning needs to be done before calling HfcVd. It is simpler this way.

Reorganisation
--------------

* test_HfcVd.py is renamed test_vd.py

Examples
--------

* The examples *Hematite drill core* and *Smokestack* was updated.
* A new one: *Convex hull*.

version 0.11.0 (beta)
=====================

This version support a noise reduction module called 'noise'. The first version of this module is composed by:

* Savitzky-Golay filter, a convolution based low-pass denoising algorithm, this algorithm can be applied on bands or on spectra;
* withening;
* MNF (maximum noise fraction), this is a first version.

To be compatible with the noise reduction module, ATGP and NFINDR needed some adjustments and they was updated accordingly:

* the internal call to HfcVd is removed, the number of endmembers to extract is now mandatory;
* a 'transform' parameter is added to NFINDR; it accept a transformed HSI cube; when a transformed cube is submited to NFINDR, the built-in PCA call is bypassed.

version 0.10.1 (beta)
=====================

This is a patch level version for the 0.10 version. The function FCLS is fixed. To run this new FCLS you need to install CVXOPT. Note that you need CVXOPT only if your intent is to run FCLS. See the Christoph Gohlke web site, it support CVXOPT for Windows.

version 0.10 (beta)
===================

* For all the classifiers classes: a new *constrained* parameter and a new *get_single_map()* method.
* For the unmixing classes NFINDR and ATGP: a new *mask* parameter
* A comprehensive exception mechanism
* Unit tests
* Two examples, one with a hematite drill core and the other with a smokestack

Reorganization
--------------

* The unmix module is renamed eea (for endmembers extraction algorithms)
* The unmix module method 'unmix' is renamed 'extract'

fixes
-----

* classifiers.SAM, SID and NormXCorr: the current plot_single_map method accept 0 for the lib_idx parameter, the expected value is 1. -- fixed
* classifiers.NormXCorr: the colorbar is inverted and the threshold value is inverted from the expected behavior. -- fixed

version 0.09 (alpha)
====================

* The main improvement for this version is a threshold parameter added to the classifiers classes. The threshold can be single value or multiple values. In the later case, there is one individual threshold for each signal to match. Another sweet improvement to the classifiers classes is the capacity to plot one map by each signal to match with the threshold of your choice.
* A new module receives all the classes and functions that work around the USGS library and the convex hull removal. This new module name is 'spectro'. This move leaves only one function in the sigproc module. A new class add get and search functionality to the USGS library.
* Module abundance_map is renamed abundance_maps.
* Module classification is renamed classifiers.
* A new example that use the Telops Hyper-Cam instrument.

version 0.08 (alpha)
====================

The two majors improvements are a substantial speed up for NFINR and an interface to the convex_hull_removal function.

In details:

NFINDR exist in two versions now. The pure python version have a speed gain of 4x against the version 0.07. The code is the same, the only difference is an almost direct call to lapack. Something that we can do with numpy and scipy. The second version is a cython rewrite without a call to lapack. It give a 8x speedup compared to the 0.07 version. This result seems abnormal, beating MKL blas/lapack is difficult but not impossible. It depends on the context, check the unmixing module documentation.

Speedup:
 GLRT: 2x against the 0.07 version

New:
 class interface to the convex_hull_removal function (see the sigproc module)
 continuum based features extractor with a tetracorder style (see the sigproc module)
 Distutils setup

Renamed:
 files tests/ex_*.py renamed to test_*.py

Updated:
 test_hull.py

version 0.07 (alpha)
====================

The more important improvements are a documentation, a NormXCorr classifier and a fix to NFINDR.

New:
 documentation
 OSP target detection
 NormXCorr distance measure
 chebyshev distance measure
 NormXCorr classifier
 corr stat function
 cov stat function

Fixed:
 NFINDR

NFINDR have a new feature. You can use ATGP to initialize the first endmembers set.

convert2d_signal_last is renamed to convert2D
the signal module is renamed to sigproc

version 0.06 (alpha)
====================

Many improvements to UCLS, NNLS, FCLS, ACE, MatchedFilter, HfcVd, NFINDR (all now in C-order). In overall this version do a better use of numpy.

Speedup improvements to ACE, SAM and SID.

New functions:
hyperCem -> detextion.CEM
hyperGLRT -> detection.GLRT
eia.FIPPI -> unmixing.FIPPI

Added a get_idx method to the classes PPI, NFINDR, ATGP, FIPPI. This method return a index list corresponding to the induced endmembers.

Added a legend to the plot function for the classes SAM and SID.

Unlike the previous version, Cython was not use for this one. Exploring the power of numpy gave very good speedup to SID and SAM.

Fix: NFINDR: np.zeros((p), dtype=np.int16) become IDX =  np.zeros((p), dtype=np.long)

In conclusion, almost all the code was revisited and improved.

version 0.05 (alpha)
====================

The library was reorganized in many modules: abundance_map, classification, detection, formatting, material_count, signal, tests, unmixing and util. The functions as been sorted into them and in some case a class interface to the function was added. See the tests examples, they are the only documentation for now.

Cython has been introduced. For this version, only the SAM classifier is compiled with cython.

Three distributions come with this version:
- the source
- one with SAM classifier compiled for Windows 7 32 bits and Python 2.7 and
- one with SAM classifier compiled for Windows 7 64 bits and Python 2.7

You can use the source package, it fall back on the python version of the SAM classifier. To use, unzip the package and place the checkout directory on your PYTHONPATH.

ex_clsf.py is renamed to ex_clf.py


For the class interfaces, the format of the HSI data cube is (m x n x p), m and n are the spatial indices and p is the pixel. And for a library of signals the format is (N x p), N is the indice and p the pixel.

version 0.04 (alpha)
====================

Note : The SPy library use the C-order for the numpy representation of the data. The signal is in the innermost loop. That's the best representation for almost all algorithms that we can apply to this kind of data. In a first attemp I tried to simulate the Matlab way of doing for hyperspectral processing (usin a F-order shape). But for this version I am revising my position and started of doing things in a more pythonic approach. For the listed functions the signal is always last. This give a gain of around 20% in speed improvement :

SAM, SID, SAM_classifier, SID_classifier

For the next versions, more functions will be migrated to the C-order, signal last.

Added from the Matlab Hyperspectral Toolbox:
hyperMatchedFilter -> MatchedFilter (with a good optimisation from numpy)
hyperAce -> ACE

A test program for the detections algorithms -> ex_detect.py

New functions, not from the current sources, but from articles : SID and SAM
and helper functions SID_classifier and SAM_classifier

A test program for the classification algorithms -> ex_clsf.py

version 0.03 (alpha)
====================

Added from the Matlab Hyperspectral Toolbox:
hyperNnls -> NNLS

From the Endmember Induction Algorithms toolbox (EIA):
EIA_ATGP -> ATGP

Patch to convert3d(): it can now accept a 1D vector as input

version 0.02 (alpha)
====================

Added from the Matlab Hyperspectral Toolbox:
hyperConvexHullRemoval -> convex_hull_removal
hyperPpi -> PPI

From the Endmember Induction Algorithms toolbox (EIA):
EIA_NFINDR -> NFINDR

Added two new examples, one for the convex_hull_removal, ex_hull.py
and one for NFIND, ex_eia.py

A ligthweight data for the tests.

Many improvements to the tests programs.

version 0.01 (alpha)
====================

First version!

From the Matlab Hyperspectral Toolbox:

hyperConvert2D -> convert2d_signal_first, convert2d_signal_last

hyperConvert3D -> convert3d

hyperNormalize -> normalize

hyperAtgp -> ATGP

hyperHfcVd -> HfcVd

hyperPct -> PCT, but use scikit.learn PCA

hyperUcls -> UCLS

hyperFcls -> FCLS

hyperCov -> use numpy.cov

hyperCorr -> use numpy.corrcoef

From the The piecewise constant toolbox:
pwc_bilateral -> bilateral

Bug fixes (old)
===============

* The N-FINDR algorithm have a problem that make it unusable. This is true for version 0.06 and previous. The pixel that give an expansion of the simplex was not saved and NFINDR never converge to a solution (2013-11-09).

* mht.ATPG signal a singular matrix and exit when the end members asked is more than, say, 30. It's a random bug. Fix: this function is deprecated, it was replaced with the eia ATGP (2013-08-05)
