import os
import sys
import tempfile

import pandas as pd
import numpy as np

import numerox as nx
from numerox.data import ERA_STR_TO_FLOAT, REGION_STR_TO_FLOAT

TEST_DATA = os.path.join(os.path.dirname(__file__), 'tests', 'test_data.hdf')


def assert_data_equal(obj1, obj2, msg=None):
    "Assert that two data (or prediction) objects are equal"
    try:
        pd.testing.assert_frame_equal(obj1.df, obj2.df)
    except AssertionError as e:
        # pd.testing.assert_frame_equal doesn't take an error message as input
        if msg is not None:
            msg = '\n\n' + msg + '\n\n' + e.args[0]
            e.args = (msg,)
        raise


def shares_memory(data1, data_or_array2):
    "True if `data1` shares memory with `data_or_array2`; False otherwise"

    isdata_like = isinstance(data_or_array2, nx.Data)
    isdata_like = isdata_like or isinstance(data_or_array2, nx.Prediction)

    if hasattr(data1, 'column_list'):
        cols = data1.column_list()
    else:
        cols = data1.pairs(as_str=False)
    cols += ['ids']

    for col in cols:
        if col == 'ids':
            a1 = data1.df.index.values
        else:
            a1 = data1.df[col].values
        if isdata_like:
            if col == 'ids':
                a2 = data_or_array2.df.index.values
            else:
                if col not in data_or_array2.df:
                    continue
                a2 = data_or_array2.df[col].values
        else:
            a2 = data_or_array2
        if np.shares_memory(a1, a2):
            return True
    return False


def micro_data(index=None):
    "Returns a tiny data object for use in unit testing"
    cols = ['era', 'region', 'x1', 'x2', 'x3',
            'bernie', 'elizabeth', 'jordan', 'ken', 'charles']
    df = pd.DataFrame(columns=cols)
    d0 = ['era1', 'train'] + [0.00, 0.01, 0.02] + [0., 0., 1., 1., 0.]
    d1 = ['era2', 'train'] + [0.10, 0.11, 0.12] + [1., 1., 1., 1., 0.]
    d2 = ['era2', 'train'] + [0.20, 0.21, 0.22] + [0., 1., 1., 1., 1.]
    d3 = ['era3', 'validation'] + [0.30, 0.31, 0.32] + [1., 1., 0., 1, 0.]
    d4 = ['era3', 'validation'] + [0.40, 0.41, 0.42] + [0., 0., 0., 1, 0.]
    d5 = ['era3', 'validation'] + [0.50, 0.51, 0.52] + [1., 1., 1., 1, 0.]
    d6 = ['era4', 'validation'] + [0.60, 0.61, 0.62] + [0., 1., 0., 1, 1.]
    d7 = ['eraX', 'test'] + [0.70, 0.71, 0.72] + [1., 1., 1., 1, 1.]
    d8 = ['eraX', 'test'] + [0.80, 0.81, 0.82] + [0., 0., 0., 0, 0.]
    d9 = ['eraX', 'live'] + [0.90, 0.91, 0.92] + [0., 1., 0., 0, 1.]
    df.loc['index0'] = d0
    df.loc['index1'] = d1
    df.loc['index2'] = d2
    df.loc['index3'] = d3
    df.loc['index4'] = d4
    df.loc['index5'] = d5
    df.loc['index6'] = d6
    df.loc['index7'] = d7
    df.loc['index8'] = d8
    df.loc['index9'] = d9
    df['era'] = df['era'].map(ERA_STR_TO_FLOAT)
    df['region'] = df['region'].map(REGION_STR_TO_FLOAT)
    if index is not None:
        df = df.iloc[index]
    df = df.copy()  # assure contiguous memory
    data = nx.Data(df)
    return data


def micro_prediction(index=None):
    "Returns a tiny prediction object for use in unit testing"
    cols = [('model0', 2), ('model1', 1), ('model2', 3), ('model0', 5)]
    df = pd.DataFrame(columns=cols)
    df.loc['index0'] = [0.002, 0.011, 0.023, 0.005]
    df.loc['index1'] = [0.102, 0.111, 0.123, 0.105]
    df.loc['index2'] = [0.202, 0.211, 0.223, 0.205]
    df.loc['index3'] = [0.302, 0.311, 0.323, 0.305]
    df.loc['index4'] = [0.402, 0.411, 0.423, 0.405]
    df.loc['index5'] = [0.502, 0.511, 0.523, 0.505]
    df.loc['index6'] = [0.602, 0.611, 0.623, 0.605]
    df.loc['index7'] = [0.702, 0.711, 0.723, 0.705]
    df.loc['index8'] = [0.802, 0.811, 0.823, 0.805]
    df.loc['index9'] = [0.902, 0.911, 0.923, 0.905]
    if index is not None:
        df = df.iloc[index]
    df = df.copy()  # assure contiguous memory
    prediction = nx.Prediction(df)
    return prediction


def play_data():
    "About 1% of a regular Numerai dataset, so contains around 60 rows per era"
    return nx.load_data(TEST_DATA)


def update_play_data(data=None, fraction=0.01):
    "Create and save data used by play_data function"
    if data is None:
        data = nx.numerai.download_data_object()
    play = data.subsample(fraction=fraction, seed=0)
    play.save(TEST_DATA)


def create_tempfile(path):
    "Create temporary file"
    return os.path.join(tempfile.gettempdir(), path)


def delete_tempfile(path):
    "Remove file"
    try:
        os.remove(path)
    except:  # noqa
        pass


# taken from https://stackoverflow.com/a/45669280
# modified for use in numerox
class HiddenPrints(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
