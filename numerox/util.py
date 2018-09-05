import sys
import numpy as np
import pandas as pd

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


def history():
    "History of changes made to the Numerai tournaments"
    d = [
         [1, 'December 1, 2015'],
         [51, 'first live logloss'],
         [61, 'first stake; $3000 prize pool'],
         [67, 'the big burn'],
         [78, 'stake prize pool increased to $6000'],
         [81, 'originality no longer a staking requirement'],
         [85, 'rounds resolve on Saturdays instead of Mondays'],
         [94, 'main tournament dropped; staking adds nmr prizes'],
         [100, 'rank corr > 0.1 with example predictions'],
         [101, 'corr > 0.1 with example predictions'],
         [102, 'logloss benchmark 0.693; corr>0.2; [0.3, 0.7]'],
         [111, '5 tournament format'],
         [113, 'conditional staking removed; min confidence 0.1'],
         [113, 'resolve Tuesday; better corporate action adjust'],
        ]
    columns = ['round', 'comment']
    df = pd.DataFrame(data=d, columns=columns)
    df = df.set_index('round')
    return df


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
