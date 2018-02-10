.. automodule:: pysptools.classification

Classification
***************

This module supports following supervised algorithms: 

* `Abundance Classification`_
* `Normalized Cross Correlation (NormXCorr)`_
* `Spectral Angle Mapper (SAM)`_
* `Spectral Information Divergence (SID)`_
* `Output`_

For each of these classifiers, three different plotting methods are availables.
For NormXCorr, SAM and SID, the fusions of classification maps use a best score win approach.

.. seealso:: 
	See the file :download:`test_cls.py<../../tests/test_cls.py>` for an example and for SVC see :download:`test_SVC.py<../../tests/test_SVC.py>`.


Apart from these supervised algorithms, following unsupervised algorithm is also supported:

* `KMeans`_ 

.. seealso::
	See the file :download:`test_kmeans.py<../../tests/test_kmeans.py>` for an example.


Abundance Classification
========================

.. autoclass:: pysptools.classification.AbundanceClassification
    :members:

Normalized Cross Correlation (NormXCorr)
==========================================

.. autoclass:: pysptools.classification.NormXCorr
    :members:

Spectral Angle Mapper (SAM)
============================

.. autoclass:: pysptools.classification.SAM
    :members:

Spectral Information Divergence (SID)
=======================================

.. autoclass:: pysptools.classification.SID
    :members:

Output
======

.. autoclass:: pysptools.classification.Output
    :members:
