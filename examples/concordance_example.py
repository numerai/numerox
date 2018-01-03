#!/usr/bin/env python

"""
Example showing how to calculate concordance.
Concordance must be less than 0.12 to pass numerai's check.
For an accurate concordance calculation Data must be the full dataset.
"""

import numerox as nx


def concordance_example():

    data = nx.play_data()
    prediction = nx.Prediction()

    prediction['logistic'] = nx.production(nx.logistic(), data)
    prediction['mlpc'] = nx.production(nx.mlpc(), data)

    concord = nx.concordance(data, prediction)
    print("\nA concordance less than 0.12 is passing")
    print(concord)


if __name__ == '__main__':
    concordance_example()
