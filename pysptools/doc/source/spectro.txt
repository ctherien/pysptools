.. automodule:: pysptools.spectro

Spectro Function and Classes
****************************

This module supports the functions convex_hull_removal and the
classes SpectrumConvexHullQuotient, FeaturesConvexHullQuotient and USGS06SpecLib.

.. seealso::
	See the file :download:`test_hull.py<../../tests/test_hull.py>`
	for a SpectrumConvexHullQuotient and FeaturesConvexHullQuotient example. The file :download:`test_spectro.py<../../tests/test_spectro.py>` is the example for the USGS06SpecLib class.

Convex Hull Removal
===================

.. autofunction:: pysptools.spectro.convex_hull_removal



FeaturesConvexHullQuotient
==========================

.. autoclass:: pysptools.spectro.FeaturesConvexHullQuotient
    :members:



SpectrumConvexHullQuotient
==========================

.. autoclass:: pysptools.spectro.SpectrumConvexHullQuotient
    :members:


USGS06SpecLib
=============

.. autoclass:: pysptools.spectro.EnviReader
    :members:

.. autoclass:: pysptools.spectro.JSONReader
    :members:

.. autoclass:: pysptools.spectro.USGS06SpecLib
    :members:
