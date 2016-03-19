Introduction
++++++++++++

The PySptools conventions are almost the same as the Matlab Hyperspectral Toolbox of Isaac Gerg. I replicate here is own text adapted to the current situation.

Notation
--------

Hyperspectral data is often expressed many ways to better describe the mathematical handling of the data; mainly as a vector of pixels when referring to the data in a space or a matrix of pixels when referring to data as an image.

For consistency, a common notation is defined to differentiate these concepts clearly. Hyperspectral data examined like an image will be defined as a matrix Mm x n x p of dimension m x n x p where m is defined as the number of rows in the image, n is defined as the number of columns in the image, and p is defined as the number of bands in the image. Therefore, a single element of such an image will be accessed using M[i,j,k] and a single pixel of an image will be accessed using M[i,j,:]. Hyperspectral data formed as a vector of vectors (i.e. 2D matrix) is defined as M(m * n) x p of dimension (m * n) x p. A single element is accessed using M[i,j] and a single pixel is accessed using M[i,:]. However, this last notation is encapsulated in the different classes and you do not have to bother with it if you do not use directly the functions.

In Python, index start at zero. When we refer to a numbered spectrum *one* in the documentation and the plotting result, is index is 0 (and so on for the others spectra).

The list below provides a summary of the notation convention used 
throughout the code.

* E : Spectral library. Each row of the matrix represents a
  spectrum vector.
* M : Data matrix:
 * Defined as an image of spectral signatures or vectors:
  * Mm x n x p. 
 * Or, defined as a long vector of spectral signatures:
  * M(m * n) x p.
* N : The total number of pixels. For example N = m * n.
* m : Number of rows in the image.
* n : Number of columns in the image.
* p : Number of bands.
* q : Number of classes / endmembers.
* U : Matrix of endmembers. Each row of the matrix represents an
  endmember vector.

Data structure
--------------

The protocol to exchange data between the different classes is very simple: it is a numpy HSI cube or a numpy spectral library. You just need to keep them in the format (m x n x p) for the former and (n x p) for the later.

Some plot functions need informations for their axis. They are given with a dictionary *axes* that have this format:

* axes['wavelength'] : a wavelengths list (1D python list). If None or not specified the list is automaticaly numbered starting at 1.
* axes['x'] : the x axis label, 'Wavelength' if None or not specified. axes['x'] is copied verbatim.
* axes['y'] : the y axis label, 'Brightness' if None or not specified. axes['y'] is copied verbatim.

Look at the file test_eea.py for an example. Additional information will be added with future versions.

Interactivity
-------------

Pysptools was initially designed to write any results into files. Recently, a new display method was added. When in IPython, you can use this method to show the results. It opens the door to a better interactivity.

Design
------

We try to keep the structure flat. Most modules are independent and can be executed easily.