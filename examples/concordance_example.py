#!/usr/bin/env python

"""
Example showing how to calculate concordance.
Concordance must be less than 0.12 to pass numerai's check.
For an accurate concordance calculation Data must be the full dataset.
"""

import numerox as nx


def concordance_example():
    data = nx.play_data()
    model = nx.model.logistic()
    prediction = nx.production(model, data)
    concord = nx.concordance(data, prediction)
    print("concordance {:.4f} (less than 0.12 is passing)".format(concord))


if __name__ == '__main__':
    concordance_example()
