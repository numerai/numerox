#!/usr/bin/env python

"""
Example of how to prepare a submission for the Numerai tournament.
It uses Numerox which you can install with: pip install numerox
For more information see: https://github.com/kwgoodman/numerox
"""

import numerox as nx


def main():

    # download dataset from numerai
    nx.download_dataset('numerai_dataset.zip', verbose=True)

    # load numerai dataset
    data = nx.load_zip('numerai_dataset.zip', verbose=True)

    # we will use logistic regression; you will want to write your own model
    model = nx.model.logistic()

    # fit model with train data and make predictions for tournament data
    prediction = nx.production(model, data)

    # save predictions to csv file for later upload to numerai
    prediction.to_csv('logistic.csv', verbose=True)


"""
Typical output:

Download dataset numerai_dataset.zip
region    train, validation, test, live
rows      884545
era       98, [era1, eraX]
x         50, min 0.0000, mean 0.4993, max 1.0091
y         mean 0.499961, fraction missing 0.3109
logistic(inverse_l2=1e-05)
      logloss   auc     acc     ystd
mean  0.692993  0.5157  0.5115  0.0028  |  region   validation
std   0.000235  0.0234  0.0179  0.0000  |  eras     12
min   0.692440  0.4853  0.4886  0.0028  |  consis   0.7500
max   0.693330  0.5734  0.5555  0.0028  |  75th     0.6931
Save logistic.csv
"""

if __name__ == '__main__':
    main()
