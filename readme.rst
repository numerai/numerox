.. image:: https://travis-ci.org/numerai/numerox.svg?branch=master
    :target: https://travis-ci.org/numerai/numerox
.. image:: https://ci.appveyor.com/api/projects/status/github/kwgoodman/numerox?svg=true&passingText=passing&failingText=failing&pendingText=pending
    :target: https://ci.appveyor.com/project/kwgoodman/numerox
.. image:: https://img.shields.io/pypi/v/numerox.svg
   :target: https://pypi.python.org/pypi/numerox/
numerox
=======

Numerox is a Numerai tournament toolbox written in Python.

All you have to do is create a model. Take a look at `model`_ for examples.

Once you have a model numerox will do the rest. First download the Numerai
dataset and then load it::

    >>> import numerox as nx
    >>> data = nx.download('numerai_dataset.zip')

Let's use the logistic regression model in numerox to run 5-fold cross
validation on the training data::

    >>> model = nx.logistic()
    >>> prediction = nx.backtest(model, data, tournament='bernie', verbosity=1)
    logistic(inverse_l2=0.0001)
           logloss     auc     acc    ystd   stats
    mean  0.692885  0.5165  0.5116  0.0056   tourn  bernie
    std   0.000536  0.0281  0.0215  0.0003  region   train
    min   0.691360  0.4478  0.4540  0.0050    eras     120
    max   0.694202  0.5944  0.5636  0.0061  consis   0.625

OK, results are good enough for a demo so let's make a submission file for the
tournament. We will fit the model on the train data and make our predictions
for the tournament data::

    >>> prediction = nx.production(model, data, 'bernie', verbosity=1)
    logistic(inverse_l2=0.0001)
           logloss     auc     acc    ystd   stats
    mean  0.692808  0.5194  0.5142  0.0063   tourn      bernie
    std   0.000375  0.0168  0.0137  0.0001  region  validation
    min   0.691961  0.4903  0.4925  0.0062    eras          12
    max   0.693460  0.5553  0.5342  0.0064  consis        0.75

Let's upload our predictions to enter the tournament::

    >>> prediction.to_csv('logistic.csv')
    >>> upload_id, status = nx.upload('logistic.csv', 'bernie',
                                      public_id, secret_key, model_id)
    metric                  value   minutes
    concordance              True   0.0898
    consistency              0.75   0.0898
    originality             False   0.1783
    validation_logloss     0.6928   0.1783
    stakeable                True   0.1783

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

Requirements: numpy, scipy, pandas, sklearn, pytables, numerapi,
setuptools, requests, nose.

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
Where indicated by code comments parts of NumPy are included in numerox. The
NumPy license appears in the licenses directory.


.. _model: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/model.rst
.. _examples: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/readme.rst
.. _chat: https://community.numer.ai/channel/numerox
.. _bugs: https://github.com/kwgoodman/numerox/issues
.. _what's new: https://github.com/kwgoodman/numerox/blob/master/release.rst
.. _Numerai: https://numer.ai
