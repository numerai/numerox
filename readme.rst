.. image:: https://travis-ci.org/kwgoodman/numerox.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/numerox
.. image:: https://ci.appveyor.com/api/projects/status/github/kwgoodman/numerox?svg=true&passingText=passing&failingText=failing&pendingText=pending
    :target: https://ci.appveyor.com/project/kwgoodman/numerox
.. image:: https://img.shields.io/pypi/v/numerox.svg
   :target: https://pypi.python.org/pypi/numerox/
numerox
=======

Numerox is a Numerai tournament toolbox written in Python.

All you have to do is create a model. Take a look at `model.py`_ for examples.

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
    >>> prediction2.to_csv('logistic.csv')  # 6 decimal places by default

There is no overlap in ids between prediction1 (train) and prediction2
(tournament) so you can add (concatenate) them if you're into that and let's
go ahead and save the result::

    >>> prediction = prediction1 + prediction2
    >>> prediction.save('logloss_1e-05.pred')  # HDF5

Once you have run and saved several predictions, you can make a report::

    >>> report = nx.report.load_report('/round79', extension='pred')
    >>> report.performance(data['train'], sort_by='logloss')
    logloss   auc     acc     ystd    consis (train; 85 eras)
    0.692455  0.5215  0.5149  0.0219  0.6824        logistic_1e-03
    0.692487  0.5224  0.5159  0.0121  0.7294        logistic_1e-04
    0.692565  0.5236  0.5162  0.0086  0.7294  extratrees_nfeature7
    0.692581  0.5206  0.5143  0.0253  0.6000        logistic_1e-02
    0.692629  0.5240  0.5164  0.0074  0.7294  extratrees_nfeature5
    0.692704  0.5200  0.5140  0.0273  0.5412        logistic_1e-01
    0.692747  0.5232  0.5162  0.0055  0.7647  extratrees_nfeature3
    0.692831  0.5238  0.5163  0.0042  0.7647  extratrees_nfeature2
    0.692974  0.5226  0.5159  0.0023  0.7647        logistic_1e-05

The lowest logloss on the train data was by ``logistic_1e-03``. Let's look at
its per era performance on the validation data::

    >>> report.performance_per_era(data['validation'], 'logistic_1e-03')
    logistic_1e-03
           logloss   auc     acc     ystd  
    era86  0.691499  0.5322  0.5296  0.0220
    era87  0.689715  0.5552  0.5371  0.0219
    era88  0.692501  0.5189  0.5167  0.0220
    era89  0.694544  0.4954  0.4916  0.0218
    era90  0.691133  0.5349  0.5230  0.0221
    era91  0.692794  0.5140  0.5061  0.0218
    era92  0.694579  0.4933  0.4906  0.0217
    era93  0.694098  0.4983  0.4954  0.0218
    era94  0.688417  0.5752  0.5591  0.0218
    era95  0.691734  0.5265  0.5224  0.0216
    era96  0.693184  0.5119  0.5092  0.0215
    era97  0.693276  0.5077  0.5089  0.0215

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

But that is slow (~9 seconds) which is painful for dedicated overfitters.
Let's create an HDF5 archive::

    >>> data.save('numerai_dataset.hdf')
    >>> data2 = nx.load_data('numerai_dataset.hdf')

That loads quickly (~0.1 seconds, but takes more disk space than the
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

    >>> nx.concat_data([data['live'], data['era1'], data['era92']])
    region    live, train, validation
    rows      19194
    era       3, [era1, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean 0.499960, fraction missing 0.3544

To get views (not copies) of the data as numpy arrays use ``data.ids``,
``data.x``, ``data.y``. To get copies (not views) of era and region as numpy
string arrays use ``data.era``, ``data.region``. Internally era and region are
stored as floats. To get views: ``data.era_float``, ``data.region_region``.

Numerox comes with a small dataset to play with::

    >>> nx.play_data()
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

Resources
=========

- Ask usage questions `on rocket.chat`_
- Report bugs `on github`_
- Check out the `release notes`_ to see what is new

Sponsor
=======

Thank you Numerai for providing funding towards the development of Numerox.

License
=======

Numerox is distributed under the the GPL v3+. See LICENSE file for details.


.. _model.py: https://github.com/kwgoodman/numerox/blob/master/numerox/model.py 
.. _at github: https://github.com/kwgoodman/numerox/issues
.. _on rocket.chat: https://community.numer.ai/channel/numerox
.. _on github: https://github.com/kwgoodman/numerox
.. _release notes: https://github.com/kwgoodman/numerox/blob/master/release.rst
