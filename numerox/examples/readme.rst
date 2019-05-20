Numerox examples
================

Numerox is a Numerai tournament toolbox written in Python.

Numerox contains three main classes. **Data** holds the Numerai dataset, parts
of which are passed to a **Model** which makes a **Prediction** that is stored
and analyzed.

- `Data`_
- `Model`_
- `Prediction`_

Running your model involves passing data to it and collecting its predictions,
tasks that numerox automates.

- `Run model`_

If you don't have time to `port your model`_ to numerox you can `import
predictions`_ from a Numerai-style csv file and do all kinds of analysis on it.

If you're the kind of person who prefers to browse example code then have a
look at these:

- `production.py`_
- `backtest.py`_
- `concordance.py`_
- `improve_model.py`_

Here's some miscellaneous stuff:

- Numerai's `CV warning`_  to hold out eras not rows
- `Stake information`_

You can run all the examples [1]_::

    >>> import numerox as nx
    >>> nx.examples.run_all_examples()

.. [1] The first_tournament example is skipped because it writes to disk.

.. _data: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/data.rst
.. _model: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/model.rst
.. _prediction: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/prediction.rst

.. _run model: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/run.rst

.. _port your model: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/model.rst
.. _import predictions: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/prediction.rst

.. _production.py: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/production.py
.. _backtest.py: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/backtest.py
.. _concordance.py: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/concordance.py
.. _improve_model.py: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/improve_model.py

.. _cv warning: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/cv_warning.rst
.. _stake information: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/show_stakes.rst
