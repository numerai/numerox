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
    region    live, test, train, validation
    rows      884544
    era       98, [era1, eraX]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.3109

Let's use the logistic regression model in numerox to run 5-fold cross
validation on the training data::

    >>> model = nx.model.logistic()
    >>> prediction = nx.backtest(model, data, verbosity=1)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.692974  0.5226  0.5159  0.0023  |  region   train
    std   0.000224  0.0272  0.0205  0.0002  |  eras     85
    min   0.692360  0.4550  0.4660  0.0020  |  consis   0.7647
    max   0.693589  0.5875  0.5606  0.0027  |  75th     0.6931

OK, results are good enough for a demo so let's make a submission file for the
tournament. We will fit the model on the train data and make our predictions
for the tournament data::

    >>> prediction = nx.production(model, data)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.692993  0.5157  0.5115  0.0028  |  region   validation
    std   0.000225  0.0224  0.0172  0.0000  |  eras     12
    min   0.692440  0.4853  0.4886  0.0028  |  consis   0.7500
    max   0.693330  0.5734  0.5555  0.0028  |  75th     0.6931
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
