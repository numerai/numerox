import numerox as nx


def compare_models(data):
    """
    Run multiple models: fit on training data, predict for tournament data.
    Then compare performance of the models
    """

    tourn = 1

    # we'll look at 5 models
    prediction = nx.production(nx.logistic(), data, tourn, verbosity=1)
    prediction += nx.production(nx.extratrees(), data, tourn, verbosity=1)
    prediction += nx.production(nx.randomforest(), data, tourn, verbosity=1)
    prediction += nx.production(nx.mlpc(), data, tourn, verbosity=1)
    prediction += nx.production(nx.logisticPCA(), data, tourn, verbosity=1)

    # correlation of models with logistic regression
    print('\nCorrelation:\n')
    prediction.correlation('logistic_t1')

    # compare performance of models
    print('\nPerformance comparison:\n')
    print(prediction.performance(data['validation'], tourn))

    # dominance of models
    print('\nModel dominance:\n')
    print(prediction.dominance(data['validation'], tourn))

    # dominace between two models
    print('\nModel dominance between two models:\n')
    p2 = prediction[['logistic_t1', 'logisticPCA_t1']]
    df = p2.dominance(data['validation'], tourn)
    print(df)

    # concordance
    print('\nConcordance:\n')
    print(prediction.concordance(data))
