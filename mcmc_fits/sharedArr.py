#!/usr/bin/env python
# -*- coding: utf-8 -*-

def crSharedArr(arr, copy=True):
    """
    Create a shared array with the same type and shape, and optionally copy it.
    In the case of an hdf5 dataset, the function will return an error if the
    hdf5 dataset is a structured type. In this very particular case, you need
    to pass the underying numpy array, forcing you to actually read it from the
    file.

    arr : a numpy array or a hdf5 dataset
    copy: Default True

    """
    import numpy as np
    from multiprocessing import Array

    # test if we have a numpy array of structured dtype
    if arr.dtype.names is None:
        # normal array
        base = Array(arr.dtype.char, int(arr.size), lock=False)
        new = np.ctypeslib.as_array(base)
    else:
        # structured array
        # I have to allocate the same amount of bytes of arr and then covert it
        # AFTER I transformed it into a numpy array
        nbytes = arr.nbytes

        # I'm implicitly assuming here that 'b' produces 1 byte
        # https://docs.python.org/3/library/array.html#module-array
        # I put a check on this later
        base = Array('b', nbytes, lock=False)
        new = np.ctypeslib.as_array(base)

        # check that we allocated the right amount of space
        assert new.nbytes == nbytes

        # convert the new array to the right datatype
        new = new.view(arr.dtype)


    new.shape = arr.shape

    if copy:
        new[:] = arr
    return new
