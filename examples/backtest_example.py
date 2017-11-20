#!/usr/bin/env python

"""
A simple cross validation run on the training data using logistic regression
"""

import numerox as nx
from numerox.splitter import CVSplitter


def backtest_example(data_filename_hdf):
    data = nx.load_data(data_filename_hdf)
    model = nx.model.logistic()
    prediction = nx.backtest(model, data)


if __name__ == '__main__':
    backtest_example('/data/ni/ni.h5')
