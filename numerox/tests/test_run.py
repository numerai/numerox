import numerox as nx
from numerox import testing
from numerox.model import fifty


def test_run():
    "Make sure run runs"
    d = testing.play_data()
    models = [nx.logistic(), fifty()]
    splitters = [nx.TournamentSplitter(d),
                 nx.ValidationSplitter(d),
                 nx.CheatSplitter(d),
                 nx.CVSplitter(d, kfold=2),
                 nx.SplitSplitter(d, fit_fraction=0.5)]
    for model in models:
        for splitter in splitters:
            nx.run(model, splitter, tournament=2, verbosity=0)
            nx.run(model, splitter, tournament='bernie', verbosity=0)


def test_backtest_production():
    "Make sure backtest and production run"
    d = testing.micro_data()
    model = fifty()
    with testing.HiddenPrints():
        for verbosity in (0, 1, 2, 3):
            nx.backtest(model, d, tournament=3, kfold=2, verbosity=verbosity)
            nx.production(model, d, tournament='ken', verbosity=verbosity)
            nx.production(model, d, tournament=4, verbosity=verbosity)
            if verbosity == 3:
                nx.production(model, d, tournament=5, name='test',
                              verbosity=verbosity)
                nx.production(model, d, tournament='charles', name='test',
                              verbosity=verbosity)
