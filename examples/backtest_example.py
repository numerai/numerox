#!/usr/bin/env python

"""
A simple cross validation run on the training data using logistic regression
"""

import numerox as nx


def backtest_example(data):
    model = nx.logistic()
    prediction = nx.backtest(model, data)  # noqa


if __name__ == '__main__':
    data = nx.play_data()
    backtest_example(data)
