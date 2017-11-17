numerox run
===========

Both the ``production`` and ``backtest`` functions are just very thin wrappers
around the ``run`` function::

    >>> prediction = nx.run(model, splitter, verbosity=2)

where ``splitter`` iterates through fit, predict splits of the data. Numerox
comes with five splitters:

- ``tournament_splitter`` fit: train; predict: tournament (production)
- ``validation_splitter`` fit: train; predict validation
- ``cheat_splitter`` fit: train+validation; predict tournament
- ``cv_splitter`` k-fold cross validation across train eras (backtest)
- ``split_splitter`` single split of train data across eras

For example, here's how you would reproduce the ``backtest`` function::

    >>> splitter = nx.cv_splitter(data, kfold=5, seed=0)
    >>> prediction = nx.run(model, splitter)

and the ``production`` function::

    >>> splitter = nx.tournament_splitter(data)
    >>> prediction = nx.run(model, splitter)
