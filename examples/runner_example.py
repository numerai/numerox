#!/usr/bin/env python

"""
Run multiple models through simple cross validation on the training data.
Then compare performance of the models
"""

import tempfile
import shutil
import numerox as nx


def runner_example(save_dir, data):

    # we'll use cross validation over the training data
    splitter = nx.CVSplitter(data)

    # we'll look at 5 models
    m1 = {'model': nx.logistic(), 'prediction_file': 'logistic.pred'}
    m2 = {'model': nx.extratrees(), 'prediction_file': 'extratrees.pred'}
    m3 = {'model': nx.randomforest(), 'prediction_file': 'randomforest.pred'}
    m4 = {'model': nx.mlpc(), 'prediction_file': 'mlpc.pred'}
    m5 = {'model': nx.logisticPCA(), 'prediction_file': 'logisticPCA.pred'}
    run_list = [m1, m2, m3, m4, m5]

    # run all models and save results
    runner = nx.Runner(run_list, splitter, save_dir, verbosity=1)
    runner.run()

    # Alternatively:
    #
    # Instead of using splitter, run_list, and nx.Runner, you can do:
    #
    #     prediction = nx.backtest(nx.logistic(), data)
    #     prediction.save(filename)
    #
    # And similarly for the other models

    # load report
    report = nx.load_report(save_dir)

    # correlation of models with logistic regression
    print('\nCorrelation:\n')
    report.correlation('logistic')

    # compare performance of models
    print('\nPerformance comparison:\n')
    report.performance(data, sort_by='logloss')

    # dominance of models
    print('\nModel dominance:\n')
    report.dominance(data, sort_by='logloss')


if __name__ == '__main__':
    data = nx.numerai.download_data_object(verbose=True)
    data = data['train']
    save_dir = tempfile.mkdtemp()
    runner_example(save_dir, data)
    shutil.rmtree(save_dir)
