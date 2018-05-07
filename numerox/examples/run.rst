numerox run
===========

Both the ``production`` and ``backtest`` functions are just very thin wrappers
around the ``run`` function::

    >>> prediction = nx.run(model, splitter, verbosity=2)

where ``splitter`` iterates through fit, predict splits of the data. Numerox
comes with eight splitters:

- ``TournamentSplitter`` fit: train; predict: tournament (production)
- ``ValidationSplitter`` fit: train; predict validation
- ``CheatSplitter`` fit: train+validation; predict tournament
- ``CVSplitter`` k-fold cross validation across train eras (backtest)
- ``LoocvSplitter`` leave one (era) out cross validation across train eras
- ``SplitSplitter`` single fit-predict split (across eras) of data
- ``IgnoreEraCVSplitter`` traditional k-fold cross validation ignoring eras
- ``RollSplitter`` roll forward making fit-predict splits from consecutive eras

For example, here's how you would reproduce the ``backtest`` function::

    >>> splitter = nx.CVSplitter(data, kfold=5, seed=0)
    >>> prediction = nx.run(model, splitter)

and the ``production`` function::

    >>> splitter = nx.TournamentSplitter(data)
    >>> prediction = nx.run(model, splitter)
