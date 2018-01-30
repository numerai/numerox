Numerox examples
================

Numerox is a Numerai tournament toolbox written in Python.

Main classes
------------

**Data** holds the Numerai dataset parts of which are passed to a **Model**
which makes a **Prediction** that is stored and analyzed.

- `Data`_
- `Model`_
- `Prediction`_

Run model
---------

Running your model involves passing data to it and collecting its predictions,
tasks that numerox automates.

- Your `first tournament`_
- `Backtest`_ example
- The `run and splitter`_ functions

Performance
-----------

Is your model any good?

- `Compare model`_ performances
- Compare performance of a `single change`_ across several models

Miscellaneous
--------------

- `Transform features`_
- Calculate `concordance`_
- Numerai's `CV warning`_  to hold out eras not rows
- `Stake information`_

Run examples
------------

You can run all the examples [1]_::

    >>> import numerox as nx
    >>> nx.examples.run_all_examples()

.. [1] The first_tournament example is skipped because it writes to disk.

.. _data: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/data.rst
.. _model: https://github.com/kwgoodman/numerox/blob/master/numerox/model.py
.. _prediction: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/prediction.rst

.. _first tournament: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/first_tournament.py
.. _backtest: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/backtest_example.py
.. _run and splitter: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/run.rst

.. _compare model: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/compare_models.rst
.. _single change: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/compare_change.py

.. _Transform features: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/transform.rst
.. _concordance: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/concordance_example.py
.. _cv warning: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/cv_warning.rst
.. _stake information: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/show_stakes.rst
