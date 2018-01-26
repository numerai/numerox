import numpy as np
import numerox as nx


def compare_change(data):
    """
    Run multiple models: fit on training data, predict for tournament data.
    Then change the data, rerun and compare performance with and without the
    change.
    """

    print('\nStandard dataset:\n')

    # we'll look at 5 models
    prediction = nx.production(nx.logistic(), data, verbosity=1)
    prediction += nx.production(nx.extratrees(), data, verbosity=1)
    prediction += nx.production(nx.randomforest(), data, verbosity=1)
    prediction += nx.production(nx.mlpc(), data, verbosity=1)
    prediction += nx.production(nx.logisticPCA(), data, verbosity=1)

    # let's now make a change, could be anything; as an example let's add
    # the square of each feature to the dataset
    x = np.hstack((data.x, data.x * data.x))
    data2 = data.xnew(x)

    print('\nDataset expanded with squared features:\n')

    # rerun all models with the new expanded data
    prediction2 = nx.production(nx.logistic(), data2, verbosity=1)
    prediction2 += nx.production(nx.extratrees(), data2, verbosity=1)
    prediction2 += nx.production(nx.randomforest(), data2, verbosity=1)
    prediction2 += nx.production(nx.mlpc(), data2, verbosity=1)
    prediction2 += nx.production(nx.logisticPCA(), data2, verbosity=1)

    # compare performance
    print('\nCompare (1 is regular dataset; 2 expanded dataset):\n')
    print(prediction.compare(data['validation'], prediction2))
