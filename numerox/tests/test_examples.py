import numerox as nx
from numerox import testing


def test_examples():
    data = nx.play_data()
    with testing.HiddenPrints():
        nx.examples.run_all_examples(data)
