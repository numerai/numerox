import numerox as nx


def backtest(data, tournament='bernie'):
    "Simple cross validation on training data using logistic regression"
    model = nx.logistic()
    prediction = nx.backtest(model, data, tournament)  # noqa
