.. automodule:: pysptools.spectro

Spectro function and classes
****************************

This module supports the functions convex_hull_removal and the
classes SpectrumConvexHullQuotient, FeaturesConvexHullQuotient and USGS06SpecLib.

See the file :download:`test_hull.py<../tests/test_hull.py>`
for a SpectrumConvexHullQuotient and FeaturesConvexHullQuotient example. The file :download:`test_spectro.py<../tests/test_spectro.py>` is the example for the USGS06SpecLib class.

.. warning:: As version 0.13.3, the package PySptools run on both Python version 2.7 and 3.x. It don't need again the JSON patch to use the USGS library. This fonctionality is deprecated and partly removed. The JSON version of the USGS library will remain for future PySptools versions.

Deprecated
==========

.. note:: The ENVI file reader is not available for Python 3.3 and make the USGS library reader useless. To turn around the problem you can use a JSON version of the USGS library. Following is the explaination on how to do this.

To run the USGS06SpecLib class on Python 3.3 you need a JSON version of the library:

* Step on a Python 2.7 installation.
* Download the ENVI version of the library.
* Run the program USGS_ENVI_to_JSON.py on it. You can find the program in the util folder.
* And use the class spectro.JSONReader to load the JSON file, see test_hull.py for an example.

End deprecated
==============

------------------------------

FeaturesConvexHullQuotient
==========================

.. autoclass:: pysptools.spectro.FeaturesConvexHullQuotient
    :members:

------------------------------

SpectrumConvexHullQuotient
==========================

.. autoclass:: pysptools.spectro.SpectrumConvexHullQuotient
    :members:

------------------------------

convex_hull_removal
===================

.. autofunction:: pysptools.spectro.convex_hull_removal

------------------------------

USGS06SpecLib
=============

.. autoclass:: pysptools.spectro.EnviReader
    :members:

.. autoclass:: pysptools.spectro.JSONReader
    :members:

.. autoclass:: pysptools.spectro.USGS06SpecLib
    :members:
