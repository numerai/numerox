.. image:: https://travis-ci.org/kwgoodman/numerox.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/numerox
numerox
=======

Numerox is a Numerai competition toolbox written in Python.

All you have to do is create a model. Take a look at ``model.py`` for examples.

Once you have a model numerox will do the rest. First download the Numerai
dataset and then load it (there is no need to unzip it)::

    >>> import numerox as nx
    >>> nx.download_dataset('numerai_dataset.zip')
    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> data
    region    live, test, train, validation
    rows      884544
    era       98, [era1, eraX]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.3109

Let's use the logistic regression model in numerox to run 5-fold cross
validation on the training data::

    >>> model = nx.model.logistic()
    >>> prediction1 = nx.backtest(model, data, verbosity=1)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.692974  0.5226  0.5159  0.0023  |  region   train
    std   0.000224  0.0272  0.0205  0.0002  |  eras     85
    min   0.692360  0.4550  0.4660  0.0020  |  consis   0.7647
    max   0.693589  0.5875  0.5606  0.0027  |  75th     0.6931

OK, results are good enough for a demo so let's make a submission file for the
tournament::

    >>> prediction2 = nx.production(model, data)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.692993  0.5157  0.5115  0.0028  |  region   validation
    std   0.000225  0.0224  0.0172  0.0000  |  eras     12
    min   0.692440  0.4853  0.4886  0.0028  |  consis   0.7500
    max   0.693330  0.5734  0.5555  0.0028  |  75th     0.6931
    >>> prediction3.to_csv('logistic.csv')  # 8 decimal places by default

There is no overlap in ids between prediction1 (train) and prediction2
(tournament) so you can add (concatenate) them if you're into that and let's
go ahead and save the result::

    >>> prediction = prediction1 + prediction2
    >>> prediction.save('logloss_1e-05.pred')  # HDF5

Once you have run and saved several predictions, you can make a report::

    >>> report = nx.report.load_report('/round79', extension='pred')
    >>> report.performance(data['train'])
    logloss   auc     acc     ystd    consis  (train)
    0.692629  0.5240  0.5164  0.0074  0.7294  extratrees_nfeature5
    0.692565  0.5236  0.5162  0.0086  0.7294  extratrees_nfeature7
    0.692831  0.5238  0.5163  0.0042  0.7647  extratrees_nfeature2
    0.692747  0.5232  0.5162  0.0055  0.7647  extratrees_nfeature3
    0.692487  0.5224  0.5159  0.0121  0.7294  logistic_1e-04
    0.692974  0.5226  0.5159  0.0023  0.7647  logistic_1e-05
    0.692581  0.5206  0.5143  0.0253  0.6000  logistic_1e-02
    0.692455  0.5215  0.5149  0.0219  0.6824  logistic_1e-03
    0.692704  0.5200  0.5140  0.0273  0.5412  logistic_1e-01

Let's sneak a peek at the performance in two of the most difficult eras in
validation::

    >>> report.performance(data.era_isin(['era92', 'era93']))
    logloss   auc     acc     ystd    consis  (validation)
    0.693353  0.4942  0.4918  0.0072  0.0000  extratrees_nfeature5
    0.693312  0.4978  0.4951  0.0084  0.0000  extratrees_nfeature7
    0.693226  0.4948  0.4940  0.0041  0.5000  extratrees_nfeature2
    0.693303  0.4922  0.4910  0.0055  0.0000  extratrees_nfeature3
    0.693582  0.4968  0.4954  0.0131  0.0000  logistic_1e-04
    0.693182  0.4968  0.4945  0.0028  0.5000  logistic_1e-05
    0.694728  0.4954  0.4940  0.0245  0.0000  logistic_1e-02
    0.694339  0.4958  0.4930  0.0217  0.0000  logistic_1e-03
    0.695001  0.4952  0.4965  0.0264  0.0000  logistic_1e-01

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

Warning
=======

This preview release has minimal unit tests coverage (yikes!) and the code
has seen little use. The next release will likely break any code you write
using numerox---the api is not yet stable. Please report any bugs or such
to https://github.com/kwgoodman/numerox/issues.

The next release will focus on bug fixes, adding unit tests, and design
tweaks.

Data class
==========

You can create a data object from the zip archive provided by Numerai::

    >>> import numerox as nx
    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> data
    region    live, test, train, validation
    rows      884544
    era       98, [era1, eraX]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.3109

But that is slow (~7 seconds) which is painful for dedicated overfitters.
Let's create an HDF5 archive::

    >>> data.save('numerai_dataset.hdf')
    >>> data2 = nx.load_data('numerai_dataset.hdf')

That loads quickly (~0.2 seconds, but takes more disk space than the
unexpanded zip archive).

Data indexing is done by rows, not columns::

    >>> data[data.y == 0]
    region    train, validation
    rows      304813
    era       97, [era1, era97]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.000000, fraction missing 0.0000

You can also index with special strings. Here are two examples::

    >>> data['era92']
    region    validation
    rows      6048
    era       1, [era92, era92]
    x         50, min 0.0308, mean 0.4993, max 1.0000
    y         mean 0.500000, fraction missing 0.0000

    >>> data['tournament']
    region    live, test, validation
    rows      348831
    era       13, [era86, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean 0.499966, fraction missing 0.7882

If you wish to extract more than one era (I hate these eras)::

    >>> data.era_isin(['era92', 'era93'])
    region    validation
    rows      12086
    era       2, [era92, era93]
    x         50, min 0.0177, mean 0.4993, max 1.0000
    y         mean 0.500000, fraction missing 0.0000

You can do the same with regions::

    >>> data.region_isin(['test', 'live'])
    region    live, test
    rows      274966
    era       1, [eraX, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean nan, fraction missing 1.0000

Or you can remove regions (or eras)::

    >>> data.region_isnotin(['test', 'live'])
    region    train, validation
    rows      609578
    era       97, [era1, era97]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.0000

You can concatenate data objects (as long as the ids don't overlap) by
adding them together. Let's add validation era92 to the training data::

    >>> data['train'] + data['era92']
    region    train, validation
    rows      541761
    era       86, [era1, era92]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499960, fraction missing 0.0000

Or, let's go crazy::

    >>> nx.concat([data['live'], data['era1'], data['era92']])
    region    live, train, validation
    rows      19194
    era       3, [era1, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean 0.499960, fraction missing 0.3544

You can pull out numpy arrays (copies, not views) like so ``data.ids``,
``data.era``, ``data.region``, ``data.x``, ``data.y``.

Numerox comes with a small dataset to play with::

    >>> nx.load_play_data()
    region    live, test, train, validation
    rows      8795
    era       98, [era1, eraX]
    x         50, min 0.0259, mean 0.4995, max 0.9913
    y         mean 0.502646, fraction missing 0.3126

It is about 1% of a regular Numerai dataset, so contains around 60 rows per
era.

Install
=======

This is what you need to run numerox:

- python
- setuptools
- numpy
- pandas
- pytables
- sklearn
- requests
- nose

Install with pipi::

    $ sudo pip install numerox

After you have installed numerox, run the unit tests (please report any
failures)::

    >>> import numerox as nx
    >>> nx.test()
    <snip>
    Ran 12 tests 0.789
    OK
    <nose.result.TextTestResult run=12 errors=0 failures=0>

Resources
=========

Questions, comments, suggestions, bugs:

- https://community.numer.ai/channel/numerox
- https://github.com/kwgoodman/numerox/issues

License
=======

Numerox is distributed under the Simplified BSD. See LICENSE file for details.
