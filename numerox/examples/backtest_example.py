import numerox as nx


def backtest_example(data):
    "Simple cross validation on training data using logistic regression"
    model = nx.logistic()
    prediction = nx.backtest(model, data)  # noqa
