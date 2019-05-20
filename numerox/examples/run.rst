Run your model
==============

To fit a model on the train data and make predictions for the tournament data::

    >>> prediction = nx.production(model, data, 'bernie', verbosity=1)
    logistic(inverse_l2=0.0001)
           logloss     auc     acc    ystd   stats
    mean  0.692808  0.5194  0.5142  0.0063   tourn      bernie
    std   0.000375  0.0168  0.0137  0.0001  region  validation
    min   0.691961  0.4903  0.4925  0.0062    eras          12
    max   0.693460  0.5553  0.5342  0.0064  consis        0.75

To run 5-fold cross validation on the training data::

    >>> model = nx.logistic()
    >>> prediction = nx.backtest(model, data, tournament='bernie', verbosity=1)
    logistic(inverse_l2=0.0001)
           logloss     auc     acc    ystd   stats
    mean  0.692885  0.5165  0.5116  0.0056   tourn  bernie
    std   0.000536  0.0281  0.0215  0.0003  region   train
    min   0.691360  0.4478  0.4540  0.0050    eras     120
    max   0.694202  0.5944  0.5636  0.0061  consis   0.625

You can optionally set the number of folds, the random seed, and
the model used in the prediction object::

    >>> p = nx.backtest(model, data, 'bernie', kfold=3, seed=13, name='mymodel')

Both the ``production`` and ``backtest`` functions are just thin wrappers
around the ``run`` function::

    >>> prediction = nx.run(model, splitter, tournament, verbosity=2)

where ``splitter`` iterates through fit, predict splits of the data. Numerox
comes with twelve splitters:

- ``TournamentSplitter`` fit: train; predict: tournament (production)
- ``FlipSplitter`` fit: validation; predict: train
- ``ValidationSplitter`` fit: train; predict validation
- ``CheatSplitter`` fit: train+validation; predict tournament
- ``CVSplitter`` k-fold cross validation across train eras (backtest)
- ``LoocvSplitter`` leave one (era) out cross validation across train eras
- ``SplitSplitter`` single fit-predict split (across eras) of data
- ``IgnoreEraCVSplitter`` traditional k-fold cross validation ignoring eras
- ``RollSplitter`` roll forward making fit-predict splits from consecutive eras
- ``ConsecutiveCVSplitter`` CV with folds that have mostly consecutive eras
- ``CustomCVSplitter`` CV over folds given by user as list of data objects
- ``CustomSplitter`` fit/predict splits given by user as list of data tuples

For example, here's how you would reproduce the ``backtest`` function::

    >>> splitter = nx.CVSplitter(data, kfold=5, seed=0, train_only=True)
    >>> prediction = nx.run(model, splitter, tournament)

and the ``production`` function::

    >>> splitter = nx.TournamentSplitter(data)
    >>> prediction = nx.run(model, splitter, tournament)

Run multiple models through multiple tournaments
------------------------------------------------

Run a single model for a single tournament::

    >>> p = nx.production(nx.logistic(), data, tournament='bernie')

Two ways to run a single model through all five tournaments::

    >>> p = nx.production(nx.logistic(), data)
    >>> p = nx.production(nx.logistic(), data, tournament=None)

Run a single model through arbitrary list of tournaments::

    >>> p = nx.production(nx.logistic(), data, ['bernie', 'charles'])

Run multiple models through a single tournament::

    >>> p = nx.production([nx.logistic(), nx.randomforest()], data, 'bernie')

Run multiple models through all tournaments::

    >>> p = nx.production([nx.logistic(), nx.randomforest()], data)

And so on.

Model names must be unique. This will raise a ``ValueError``::

    >>> models = [nx.randomforest(depth=4), nx.randomforest(depth=5)]
    >>> p = nx.production(models, data)
    ValueError: `model` cannot contain duplicate names

Instead rename the models::

    >>> models = [nx.randomforest(depth=4).rename('rf_d4'),
                  nx.randomforest(depth=5).rename('rf_d5')]
    >>> p = nx.production(models, data)
    >>> p
          bernie elizabeth jordan ken charles
    rf_d4      x         x      x   x       x
    rf_d5      x         x      x   x       x

Tournaments must be unique. This will raise a ``ValueError``::

    >>> p = nx.production(nx.logistic(), data, tournament=[1, 'bernie'])
    ValueError: `tournament` cannot contain duplicates
