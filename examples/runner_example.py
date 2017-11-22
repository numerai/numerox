#!/usr/bin/env python

"""
Run multiple models through simple cross validation on the training data.
"""

import tempfile
import numerox as nx


def runner_example():

    # download data
    with tempfile.NamedTemporaryFile() as temp:
        print("download data from numerai")
        nx.download_dataset(temp.name)
        data = nx.load_zip(temp.name)

    # let's do a CV on train data
    splitter = nx.CVSplitter(data)

    # let's run 3 models
    m1 = {'model': nx.model.logistic(),
          'prediction_file': None,
          'csv_file': None}
    m2 = {'model': nx.model.logistic(1e-4)}
    m3 = {'model': nx.model.extratrees()}
    run_list = [m1, m2, m3]

    # we won't save anything, just display the results
    runner = nx.Runner(run_list, splitter, verbosity=1)
    runner.run()


if __name__ == '__main__':
    runner_example()
