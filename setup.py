from distutils.core import setup


setup(name = "pysptools",
    version = "0.14.2",
    description = "A hyperspectral imaging tools box",
    author = "Christian Therien",
    author_email = "ctherien@users.sourceforge.net",
    url = "http://pysptools.sourceforge.net/",
    license = "Apache License Version 2.0",
    keywords = "python, hyperspectral imaging, signal processing, library, endmembers, unmixing, pysptools, sam, sid, atgp, N-FINDR, NFINDR, spectroscopy, target detection, georessources, geoimaging, chemical imaging, pharmaceutical, pharma, minerals, spectral, remote sensing",
    packages=[  'pysptools',
                'pysptools/abundance_maps',
                'pysptools/classification',
                'pysptools/detection',
                'pysptools/distance',
                'pysptools/doc',
                'pysptools/doc/source',
                'pysptools/doc/source/bur',
                'pysptools/doc/source/chull',
                'pysptools/doc/source/hem',
                'pysptools/doc/source/pic',
                'pysptools/doc/source/smk',
                'pysptools/doc/source/_templates',
                'pysptools/eea',
                'pysptools/examples',
                'pysptools/material_count',
                'pysptools/noise',
                'pysptools/tests',
                'pysptools/sigproc',
                'pysptools/skl',
                'pysptools/spectro',
                'pysptools/util'],
    package_data={'pysptools': ['*.txt'],
                  'pysptools/doc': ['*.*'],
                  'pysptools/doc': ['*'],
                  'pysptools/doc/source': ['*.*'],
                  'pysptools/doc/source/bur': ['*.*'],
                  'pysptools/doc/source/chull': ['*.*'],
                  'pysptools/doc/source/hem': ['*.*'],
                  'pysptools/doc/source/pic': ['*.*'],
                  'pysptools/doc/source/smk': ['*.*'],
                  'pysptools/doc/source/_templates': ['*.*'],
                  'pysptools/examples': ['*.html'],
    			   'pysptools/eea': ['*.*']},
    long_description = """
PySptools is a hyperspectral and spectral imaging library that provides spectral algorithms for the Python programming language. Specializations of the library are the endmembers extraction, unmixing process, supervised classification, target detection, noise reduction, convex hull removal and features extraction at spectrum level.

The library is designed to be easy to use and almost all functionality has a plot function to save you time with the data analysis process. The actual sources of the algorithms are the Matlab Hyperspectral Toolbox of Isaac Gerg, the pwctools of M. A. Little, the Endmember Induction Algorithms toolbox (EIA), the HySime Matlab module of José Bioucas-Dias and José Nascimento and science articles.

Starting at version 0.14.0, the library introduce a scikit-learn bridge. The bridge is partial and *alpha*.

Functionalities
***************

The functions and classes are organized by topics:

* abundance maps: FCLS, NNLS, UCLS
* classification: AbundanceClassification, NormXCorr, SAM, SID
* detection: ACE, CEM, GLRT, MatchedFilter, OSP
* distance: chebychev, NormXCorr, SAM, SID
* endmembers extraction: ATGP, FIPPI, NFINDR, PPI
* material count: HfcVd, HySime
* noise: Savitzky Golay, MNF, whiten
* sigproc: bilateral
* skl: HyperEstimatorCrossVal, HyperSVC, HyperGradientBoostingClassifier, HyperRandomForestClassifier, HyperKNeighborsClassifier, HyperLogisticRegression and others
* spectro: convex hull quotient, features extraction (tetracorder style), USGS06 lib interface
* util: load_ENVI_file, load_ENVI_spec_lib, corr, cov, plot_linear_stretch, display_linear_stretch, convert2D, convert3D, normalize, InputValidation, ROIs and others

The library do an extensive use of the numpy numeric library and can achieve good speed for some functions. The library is mature enough and is very usable even if the development is at a beta stage.

Installation
************

PySptools can run under Python 2.7 and 3.5. It is tested with these versions but can probably run under others Python versions.

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
************

    * Python 2.7 or 3.x
    * Numpy, required
    * Scipy, required
    * scikit-learn, required, version >= 0.18
    * SPy, required, version >= 0.17
    * Matplotlib, required, version 1.5.3 or less (not working with 2.0.x)
    * CVXOPT, optional, to run FCLS, version 1.1.8
    * IPython, optional, if you want to use the display feature
""",
    classifiers=[
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: GIS",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Visualization"
    ],
)
