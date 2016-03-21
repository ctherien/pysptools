Installing PySptools
********************

PySptools can run under Python 2.7 and 3.5. It has been tested for these versions but can probably run under others Python versions.

.. note:: The HSI cubes are, in general, large and the 64 bits version of Python is recommended.

The latest release is available at these download sites:

	* `pypi <https://pypi.python.org/pypi/pysptools>`_
	* `sourceforge <http://sourceforge.net/projects/pysptools/>`_ 

Manual installation
===================

To install download the sources, expand it in a directory and add the path of
the pysptools-0.xx.x directory to the PYTHONPATH system variable.

Distutils installation
======================

You can use Distutils. Expand the sources in a directory,
go to the pysptools-0.xx.x directory and at the command prompt type 'python setup.py install'.
To uninstall the library, you have to do it manually. Go to your python installation. In the
Lib/site-packages folder simply removes the associated pysptools folder and files.

Dependencies
============

    * Python 2.7 or 3.5
    * Numpy, required
    * Scipy, required
    * scikit-learn, required
    * SPy, require, version >= 0.17
    * Matplotlib, required
    * CVXOPT, optional, to run FCLS
    * IPython, optional, if you want to use the display feature

The development environment is a follow:

* The library is developed on the linux platform, with anaconda2-2.4.1 and anaconda3-2.4.1.