.. image:: https://travis-ci.org/kwgoodman/numerox.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/numerox
.. image:: https://img.shields.io/pypi/v/numerox.svg
   :target: https://pypi.python.org/pypi/numerox/
numerox
=======

Numerox is a Numerai tournament toolbox written in Python.

All you have to do is create a model. Take a look at `model.py`_ for examples.

Once you have a model numerox will do the rest. First download the Numerai
dataset and then load it::

    >>> import numerox as nx
    >>> nx.download_dataset('numerai_dataset.zip')
    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> data
    region    train, validation, test, live
    rows      637205
    era       133, [era1, eraX]
    x         50, min 0.0000, mean 0.5025, max 1.0000
    y         mean 0.499924, fraction missing 0.3095

Let's use the logistic regression model in numerox to run 5-fold cross
validation on the training data::

    >>> model = nx.model.logistic()
    >>> prediction = nx.backtest(model, data, verbosity=1)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.693103  0.5159  0.5114  0.0008  |  region   train
    std   0.000080  0.0289  0.0219  0.0000  |  eras     120
    min   0.692874  0.4384  0.4446  0.0007  |  consis   0.7000
    max   0.693323  0.5962  0.5626  0.0009  |  75th     0.6932

OK, results are good enough for a demo so let's make a submission file for the
tournament. We will fit the model on the train data and make our predictions
for the tournament data::

    >>> prediction = nx.production(model, data)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.693090  0.5178  0.5137  0.0010  |  region   validation
    std   0.000060  0.0171  0.0140  0.0000  |  eras     12
    min   0.692950  0.4891  0.4927  0.0010  |  consis   0.9167
    max   0.693192  0.5556  0.5350  0.0010  |  75th     0.6931
    >>> prediction.to_csv('logistic.csv')  # 6 decimal places by default

Examples
========

Have a look at the `examples`_.

Install
=======

Install with pip::

    $ pip install numerox

After you have installed numerox, run the unit tests (please report any
failures)::

    >>> import numerox as nx
    >>> nx.test()

Requirements: python, setuptools, numpy, pandas, pytables, sklearn, requests,
nose.

Resources
=========

- Let's `chat`_
- See `examples`_
- Check `what's new`_
- Report `bugs`_

Sponsor
=======

Thank you `Numerai`_ for funding the development of Numerox.

License
=======

Numerox is distributed under the the GPL v3+. See LICENSE file for details.
Where indicated by code comments parts of NumPy and SciPy are included in
numerox. Their licenses appear in the licenses directory.


.. _model.py: https://github.com/kwgoodman/numerox/blob/master/numerox/model.py
.. _examples: https://github.com/kwgoodman/numerox/blob/master/examples/readme.rst
.. _chat: https://community.numer.ai/channel/numerox
.. _bugs: https://github.com/kwgoodman/numerox/issues
.. _what's new: https://github.com/kwgoodman/numerox/blob/master/release.rst
.. _Numerai: https://numer.ai
