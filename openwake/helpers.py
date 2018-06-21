import numpy as np

NoneType = type(None)

def find_nearest(array, value):
    """
    Find the nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def find_index(array, value):
    """
    Find index of nearest value in an array
    """
    return (np.abs(array-value)).argmin()

def set_nan_or_inf_to_zero(array):
    """
    Sets NaN or Inf values to zero.

    This is useful when wishing to avoid zero division errors when using the
    'reduce' method.

    Returns altered array
    param array list of values to change
    """
    array[np.isinf(array) + np.isnan(array)] = 0
    return array

def set_below_abs_tolerance_to_zero(array, tolerance=1e-2):
    """
    Sets values below the given tolerance to zero.

    This is useful when wishing to avoid zero division errors when using the
    'reduce' method.

    Returns altered array
    param array list of values to change
    param tolerance value below which array elements should be reassigned to zero
    """
    array[abs(array) < tolerance] = 0
    return array

