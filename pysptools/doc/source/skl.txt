.. automodule:: pysptools.skl

Scikit-learn Interface (alpha)
******************************

This module support an interface between hyperspectral algorithms and scikit-learn.

* `Cross Validation`_
* `HyperAdaBoostClassifier`_
* `HyperBaggingClassifier`_
* `HyperExtraTreesClassifier`_
* `HyperGaussianNB`_
* `HyperGradientBoostingClassifier`_
* `HyperKNeighborsClassifier`_
* `HyperLogisticRegression`_
* `HyperRandomForestClassifier`_
* `Suppot Vector Supervised Classification (HyperSVC)`_
* `Unsupervised clustering using KMeans`_

Utility functions.

* `hyper_scale`_
* `shape_to_XY`_

.. seealso::
	See the example file :download:`nbex_skl_snow <../../examples/nbex_skl_snow.html>`
	for a use of HyperEstimatorCrossVal and HyperSVC. See :download:`test_sklearn <../../tests/test_sklearn.py>` for an example.

.. note:: 
    This is an alpha version . This module will certainly grow with time and anything can change, class name, class interface and so on.


Cross Validation
================

.. autoclass:: pysptools.skl.HyperEstimatorCrossVal
    :members:

HyperAdaBoostClassifier
=======================

.. autoclass:: pysptools.skl.HyperAdaBoostClassifier
    :members:


HyperBaggingClassifier
======================

.. autoclass:: pysptools.skl.HyperBaggingClassifier
    :members:


HyperExtraTreesClassifier
=========================

.. autoclass:: pysptools.skl.HyperExtraTreesClassifier
    :members:


HyperGaussianNB
===============

.. autoclass:: pysptools.skl.HyperGaussianNB
    :members:


HyperGradientBoostingClassifier
===============================

.. autoclass:: pysptools.skl.HyperGradientBoostingClassifier
    :members:


HyperKNeighborsClassifier
=========================

.. autoclass:: pysptools.skl.HyperKNeighborsClassifier
    :members:

HyperLogisticRegression
=======================

.. autoclass:: pysptools.skl.HyperLogisticRegression
    :members:

HyperRandomForestClassifier
===========================

.. autoclass:: pysptools.skl.HyperRandomForestClassifier
    :members:

Suppot Vector Supervised Classification (HyperSVC)
==================================================

see :download:`test_HyperSVC.py<../tests/test_HyperSVC.py>` for an example

.. autoclass:: pysptools.skl.HyperSVC
    :members:

Unsupervised clustering using KMeans
====================================

See the file :download:`test_kmeans.py<../tests/test_kmeans.py>` for an example.

.. autoclass:: pysptools.skl.KMeans
    :members:

hyper_scale
===========

.. autofunction:: pysptools.skl.hyper_scale

shape_to_XY
===========

.. autofunction:: pysptools.skl.shape_to_XY

