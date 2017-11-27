import os

import pandas as pd
import numpy as np

import numerox as nx
from numerox.data import ERA_STR_TO_FLOAT, REGION_STR_TO_FLOAT

TEST_DATA = os.path.join(os.path.dirname(__file__), 'tests', 'test_data.hdf')


def assert_data_equal(data1, data2, msg=None):
    "Assert that two data objects are equal"
    try:
        pd.testing.assert_frame_equal(data1.df, data2.df)
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
    cols = data1._column_list() + ['ids']
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
    cols = ['era', 'region', 'x1', 'x2', 'x3', 'y']
    df = pd.DataFrame(columns=cols)
    df.loc['index0'] = ['era1', 'train'] + [0.00, 0.01, 0.02] + [0.]
    df.loc['index1'] = ['era2', 'train'] + [0.10, 0.11, 0.12] + [1.]
    df.loc['index2'] = ['era2', 'train'] + [0.20, 0.21, 0.22] + [0.]
    df.loc['index3'] = ['era3', 'validation'] + [0.30, 0.31, 0.32] + [1.]
    df.loc['index4'] = ['era3', 'validation'] + [0.40, 0.41, 0.42] + [0.]
    df.loc['index5'] = ['era3', 'validation'] + [0.50, 0.51, 0.52] + [1.]
    df.loc['index6'] = ['era4', 'validation'] + [0.60, 0.61, 0.62] + [0.]
    df.loc['index7'] = ['eraX', 'test'] + [0.70, 0.71, 0.72] + [1.]
    df.loc['index8'] = ['eraX', 'test'] + [0.80, 0.81, 0.82] + [0.]
    df.loc['index9'] = ['eraX', 'live'] + [0.90, 0.91, 0.92] + [1.]
    df['era'] = df['era'].map(ERA_STR_TO_FLOAT)
    df['region'] = df['region'].map(REGION_STR_TO_FLOAT)
    if index is not None:
        df = df.iloc[index]
    df = df.copy()  # assure contiguous memory
    data = nx.Data(df)
    return data


def micro_prediction(index=None):
    d = micro_data(index)
    n = len(d)
    rs = np.random.RandomState(0)
    yhat = 0.2 * (rs.rand(n) - 0.5) + 0.5
    prediction = nx.Prediction()
    prediction.append(d.ids, yhat)
    return prediction


def play_data():
    "About 1% of a regular Numerai dataset, so contains around 60 rows per era"
    return nx.load_data(TEST_DATA)


def update_play_data(data=None, fraction=0.01):
    "Create and save data used by play_data function"
    if data is None:
        data = nx.numerai.download_data_object()
    play = data.subsample(fraction=fraction, balance=True, seed=0)
    play.save(TEST_DATA)
