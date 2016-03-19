.. automodule:: classification

Supervised classification classes
*********************************

This module supports HSI cube supervised classifiers. They are NormXCorr, SAM, SID and SVC.
For each classifier three differents plotting are availables.

For NormXCorr, SAM and SID, the fusions of classification maps use a best score win approach.

See the file :download:`test_cls.py<../tests/test_cls.py>` for an example and for SVC see
:download:`test_SVC.py<../tests/test_SVC.py>`.

------------------------------

AbundanceClassification
=======================

.. autoclass:: pysptools.classification.AbundanceClassification
    :members:

------------------------------

NormXCorr
=========

.. autoclass:: pysptools.classification.NormXCorr
    :members:

------------------------------

SAM
===

.. autoclass:: pysptools.classification.SAM
    :members:

------------------------------

SID
===

.. autoclass:: pysptools.classification.SID
    :members:

------------------------------

SVC
===

.. autoclass:: pysptools.classification.SVC
    :members:

------------------------------

Output
======

.. autoclass:: pysptools.classification.Output
    :members: