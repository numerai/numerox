#!/usr/bin/env python

"""
Example showing how to calculate concordance; must be less than 0.12 to pass.
Data must be the full dataset: do not remove any data!
"""

import tempfile
import numerox as nx


def concordance_example():
    with tempfile.NamedTemporaryFile() as temp:
        print("Download data from numerai")
        nx.download_dataset(temp.name)
        data = nx.load_zip(temp.name)
    model = nx.model.logistic()
    prediction = nx.production(model, data)
    concord = nx.concordance(data, prediction)
    print("Concordance {:.4f} (less than 0.12 is passing)".format(concord))


if __name__ == '__main__':
    concordance_example()
