#!/usr/bin/env python

"""
Run multiple models through simple cross validation on the training data.
Then compare performance of the models
"""

import numerox as nx


def compare_models(data):

    # we'll look at 5 models
    prediction = nx.backtest(nx.logistic(), data, verbosity=1)
    prediction += nx.backtest(nx.extratrees(), data, verbosity=1)
    prediction += nx.backtest(nx.randomforest(), data, verbosity=1)
    prediction += nx.backtest(nx.mlpc(), data, verbosity=1)
    prediction += nx.backtest(nx.logisticPCA(), data, verbosity=1)

    # correlation of models with logistic regression
    print('\nCorrelation:\n')
    prediction.correlation('logistic')

    # compare performance of models
    print('\nPerformance comparison:\n')
    prediction.performance(data, sort_by='logloss')

    # dominance of models
    print('\nModel dominance:\n')
    prediction.dominance(data, sort_by='logloss')

    # originality given that logistic model has already been submitted
    print('\nModel originality (versus logistic):\n')
    print(prediction.originality(['logistic']))


if __name__ == '__main__':
    data = nx.numerai.download_data_object(verbose=True)
    compare_models(data['train'])
