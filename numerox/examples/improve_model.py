import numpy as np
import numerox as nx


def improve_model(data, tournament='bernie'):
    """
    Run multiple models: fit on training data, predict for tournament data.
    Then change the data, rerun and compare performance with and without the
    change.
    """

    # we'll look at 5 models
    models = [nx.logistic(), nx.extratrees(), nx.randomforest(), nx.mlpc(),
              nx.logisticPCA()]

    print('\nStandard dataset:\n')

    # first run the base case
    prediction = nx.production(models, data, tournament, verbosity=1)

    # let's now make a change, could be anything; as an example let's add
    # the square of each feature to the dataset
    x = np.hstack((data.x, data.x * data.x))
    data2 = data.xnew(x)

    print('\nDataset expanded with squared features:\n')

    # rerun all models with the new expanded data
    prediction2 = nx.production(models, data2, tournament, verbosity=1)

    # compare performance
    print('\nCompare (1 is regular dataset; 2 expanded dataset):\n')
    print(prediction.compare(data['validation'], prediction2, tournament))
