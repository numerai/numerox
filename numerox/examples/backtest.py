import numerox as nx


def backtest(data, tournament='kazutsugi'):
    "Simple cross validation on training data using linear regression"
    model = nx.linear()
    prediction = nx.backtest(model, data, tournament)  # noqa
