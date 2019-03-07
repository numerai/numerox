from nose.tools import ok_
from nose.tools import assert_raises

import numerox as nx
from numerox import testing


def test_run():
    "Make sure run runs"
    d = testing.play_data()
    models = [nx.logistic(), nx.fifty()]
    splitters = [nx.TournamentSplitter(d),
                 nx.ValidationSplitter(d),
                 nx.CheatSplitter(d),
                 nx.CVSplitter(d, kfold=2),
                 nx.SplitSplitter(d, fit_fraction=0.5)]
    for model in models:
        for splitter in splitters:
            nx.run(model, splitter, tournament=2, verbosity=0)
            nx.run(model, splitter, tournament='bernie', verbosity=0)
            p = nx.run(model, splitter, tournament=None, verbosity=0)
            ok_(p.shape[1] == 5, 'wrong number of tournaments')
            ok_(p.tournaments() == nx.tournament_all(), 'wrong tournaments')
            p = nx.run(model, splitter, verbosity=0)
            ok_(p.shape[1] == 5, 'wrong number of tournaments')
            ok_(p.tournaments() == nx.tournament_all(), 'wrong tournaments')
    assert_raises(ValueError, nx.run, None, nx.TournamentSplitter(d))
    assert_raises(ValueError, nx.run, nx.fifty(), nx.TournamentSplitter(d), {})


def test_multiple_runs():
    "test running multiple models through multiple tournaments"

    d = testing.play_data()
    models = [nx.logistic(), nx.fifty()]

    with testing.HiddenPrints():

        p = nx.production(models, d, 'bernie')
        ok_(p.shape[1] == 2, 'wrong number of tournaments')
        p = nx.backtest(models, d, 2)
        ok_(p.shape[1] == 2, 'wrong number of tournaments')
        p = nx.run(models, nx.ValidationSplitter(d), 'ken')
        ok_(p.shape[1] == 2, 'wrong number of tournaments')

        p = nx.production(models, d)
        ok_(p.shape[1] == 10, 'wrong number of tournaments')
        p = nx.backtest(models, d)
        ok_(p.shape[1] == 10, 'wrong number of tournaments')
        p = nx.run(models, nx.ValidationSplitter(d))
        ok_(p.shape[1] == 10, 'wrong number of tournaments')

        p = nx.production(models, d, [1, 5])
        ok_(p.shape[1] == 4, 'wrong number of tournaments')
        ok_(p.tournaments() == ['bernie', 'charles'], 'wrong tournaments')
        p = nx.backtest(models, d, ['charles', 'bernie'])
        ok_(p.shape[1] == 4, 'wrong number of tournaments')
        ok_(p.tournaments() == ['bernie', 'charles'], 'wrong tournaments')
        p = nx.run(models, nx.ValidationSplitter(d), ['ken'])
        ok_(p.shape[1] == 2, 'wrong number of tournaments')
        ok_(p.tournaments() == ['ken'], 'wrong tournaments')


def test_backtest_production():
    "Make sure backtest and production run"
    d = testing.micro_data()
    model = nx.fifty()
    with testing.HiddenPrints():
        p = nx.production(model, d)
        ok_(p.shape[1] == 5, 'wrong number of tournaments')
        ok_(p.tournaments() == nx.tournament_all(), 'wrong tournaments')
        p = nx.backtest(model, d, kfold=2)
        ok_(p.shape[1] == 5, 'wrong number of tournaments')
        ok_(p.tournaments() == nx.tournament_all(), 'wrong tournaments')
        for verbosity in (0, 1, 2, 3):
            nx.backtest(model, d, tournament=3, kfold=2, verbosity=verbosity)
            nx.production(model, d, tournament='ken', verbosity=verbosity)
            nx.production(model, d, tournament=4, verbosity=verbosity)
            nx.production(model, d, tournament=None, verbosity=verbosity)
            if verbosity == 3:
                nx.production(model, d, tournament=5, verbosity=verbosity)
                nx.production(model, d, tournament='charles',
                              verbosity=verbosity)


def test_run_unique():
    "name and tournament lists must be unique"
    d = testing.micro_data()
    assert_raises(ValueError, nx.production, [nx.fifty(), nx.fifty()], d)
    assert_raises(ValueError, nx.production, nx.fifty(), d, [1, 'bernie'])
