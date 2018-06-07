import numerox as nx


def concordance_example(data, tournament=1):
    """
    Example showing how to calculate concordance.
    Concordance must be less than 0.12 to pass numerai's check.
    For an accurate concordance calculation `data` must be the full dataset.
    """
    prediction = nx.production(nx.logistic(), data, tournament)
    prediction += nx.production(nx.extratrees(), data, tournament)
    prediction += nx.production(nx.mlpc(), data, tournament)
    print("\nA concordance less than 0.12 is passing")
    print(prediction.concordance(data))
