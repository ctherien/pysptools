.. automodule:: pysptools.util

Utility Functions and Classes
*****************************

The load_ENVI_file and load_ENVI_spec_lib functions are used by the examples to load the HS cubes
that are saved in the ENVI file format. The functions corr and cov are the one
defined in the Matlab Hyperspectral Toolbox.

------------------------------

Load the data and the header from an ENVI file
==============================================

.. autofunction:: pysptools.util.load_ENVI_file

------------------------------

Load an ENVI.sli file
=====================

.. autofunction:: pysptools.util.load_ENVI_spec_lib

------------------------------

Autocorrelation
===============

.. autofunction:: pysptools.util.corr

------------------------------

Covariance
==========

.. autofunction:: pysptools.util.cov

------------------------------

Display a linear stretched RGB image
====================================

.. autofunction:: pysptools.util.display_linear_stretch

------------------------------

Plot a linear stretched RGB image
=================================

.. autofunction:: pysptools.util.plot_linear_stretch

------------------------------

Converts a 3D data cube to a 2D matrix
======================================

.. autofunction:: pysptools.util.convert2d

------------------------------

Converts a 1D or 2D matrix to a 3D data cube
============================================

.. autofunction:: pysptools.util.convert3d

------------------------------

Normalize
=========

.. autofunction:: pysptools.util.normalize

------------------------------

Manage Regions of Interests
===========================

.. autoclass:: pysptools.util.ROIs
    :members:

------------------------------


The big class that check the inputs (InputValidation)
=====================================================

.. autoclass:: pysptools.util.InputValidation
    :members:

------------------------------

Read a 3D MATLAB array
======================

.. autoclass:: pysptools.util.load_mat_file
    :members:

------------------------------


Shrink a hyperspectral image
============================

.. autoclass:: pysptools.util.shrink
    :members:
