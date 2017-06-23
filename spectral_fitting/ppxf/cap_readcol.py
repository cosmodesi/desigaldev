"""
V1.0.0: Created by Michele Cappellari on Tue Feb 25 11:39:24 2014
V1.1.0: Python 3 compatibility. MC, Oxford, 27 May 2014
V1.1.1: Allow for a scalar in usecols. MC, Oxford, 2 August 2014
V1.1.2: Use **kwargs in function call. MC, Oxford, 10 May 2017

"""

import sys
import numpy as np

#-----------------------------------------------------------------------------

def readcol(filename, **kwargs):
    """
    Tries to reproduce the simplicity of the IDL procedure READCOL.
    Given a file with some columns of strings and columns of numbers, this
    function extract the columns from a file and places them in Numpy vectors
    with the proper type:

    name, mass = readcol('prova.txt', usecols=(0, 2))

    where the file prova.txt contains the following:

    ##################
    # name radius mass
    ##################
      abc   25.   36.
      cde   45.   56.
      rdh   55    57.
      qtr   75.   46.
      hdt   47.   56.
    ##################
    
    This function is a wrapper for numpy.genfromtxt() and accepts the same input.
    See the following website for the full documentation 
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

    """
    f = np.genfromtxt(filename, dtype=None, **kwargs)

    t = type(f[0])
    if t == np.ndarray or t == np.void: # array or structured array
        f = map(np.array, zip(*f))

    # In Python 3.x all strings (e.g. name='NGC1023') are Unicode strings by defauls.
    # However genfromtxt() returns byte strings b'NGC1023' for non-numeric columns.
    # To have the same behaviour in Python 3 as in Python 2, I convert the Numpy
    # byte string 'S' type into Unicode strings, which behaves like normal strings.
    # With this change I can read the string a='NGC1023' from a text file and the
    # test a == 'NGC1023' will give True as expected.

    if sys.version >= '3':
        f = [v.astype(str) if v.dtype.char=='S' else v for v in f]

    return f

#-----------------------------------------------------------------------------
