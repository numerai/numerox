from numerox.run import run, backtest, production
from numerox.testing import play_data
from numerox.model import logistic, extratrees
from numerox.splitter import (TournamentSplitter, ValidationSplitter,
                              CheatSplitter, CVSplitter, SplitSplitter)


def test_run():
    "Make sure run runs"
    d = play_data()
    models = [logistic(), extratrees(nfeatures=2)]
    splitters = [TournamentSplitter(d),
                 ValidationSplitter(d),
                 CheatSplitter(d),
                 CVSplitter(d, kfold=2),
                 SplitSplitter(d, fit_fraction=0.5)]
    for model in models:
        for splitter in splitters:
            run(model, splitter, verbosity=0)


def test_backtest_production():
    "Make sure backtest and production run"
    d = play_data()
    model = logistic()
    backtest(model, d, kfold=2, verbosity=0)
    production(model, d, verbosity=0)
