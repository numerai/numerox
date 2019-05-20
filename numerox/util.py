import sys
import numpy as np

if sys.version_info[0] == 2:
    BASE_STRING = basestring
else:
    BASE_STRING = str  # pragma: no cover


def isint(x):
    """
    Returns True if input is an integer; False otherwise.

    Parameters
    ----------
    x : any
        Input can be of any type.

    Returns
    -------
    y : bool
        True is `x` is an integer, False otherwise.

    Notes
    -----
    A table showing what isint returns for various types:

    ========== =======
       type     isint
    ========== =======
    int          True
    np.int32     True
    np.int64     True
    float        False
    np.float32   False
    np.float64   False
    complex      False
    str          False
    bool         False

    Examples
    --------
    >>> isint(1)
    True
    >>> isint(1.1)
    False
    >>> isint(True)
    False
    >>> isint(1j)
    False
    >>> isint('a')
    False

    """
    return np.issubdtype(type(x), np.signedinteger)


def isstring(s):
    "Returns True if input is a string; False otherwise."
    return isinstance(s, BASE_STRING)


def flatten_dict(dictionary):
    "flatten nested dictionaries"
    items = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            items.extend(flatten_dict(value).items())
        else:
            items.append((key, value))
    return dict(items)


def is_none_slice(index):
    "Is the slice `index` a slice(None, None, None)? True or False."
    if index.start is not None:
        return False
    if index.stop is not None:
        return False
    if index.step is not None:
        return False
    return True
