import numerox as nx


def backtest_example(data, tournament=1):
    "Simple cross validation on training data using logistic regression"
    model = nx.logistic()
    prediction = nx.backtest(model, data, tournament)  # noqa
