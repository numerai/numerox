import numerox as nx


def compare_models(data):
    """
    Run multiple models: fit on training data, predict for tournament data.
    Then compare performance of the models
    """

    # we'll look at 5 models
    prediction = nx.production(nx.logistic(), data, verbosity=1)
    prediction += nx.production(nx.extratrees(), data, verbosity=1)
    prediction += nx.production(nx.randomforest(), data, verbosity=1)
    prediction += nx.production(nx.mlpc(), data, verbosity=1)
    prediction += nx.production(nx.logisticPCA(), data, verbosity=1)

    # correlation of models with logistic regression
    print('\nCorrelation:\n')
    prediction.correlation('logistic')

    # compare performance of models
    print('\nPerformance comparison:\n')
    print(prediction.performance(data['validation'], sort_by='logloss'))

    # dominance of models
    print('\nModel dominance:\n')
    print(prediction.dominance(data['validation'], sort_by='logloss'))

    # dominace between two models
    print('\nModel dominance between two models:\n')
    df = prediction[['logistic', 'logisticPCA']].dominance(data['validation'])
    print(df)

    # originality given that logistic model has already been submitted
    print('\nModel originality (versus logistic):\n')
    print(prediction.originality(['logistic']))

    # concordance
    print('\nConcordance:\n')
    print(prediction.concordance(data))
