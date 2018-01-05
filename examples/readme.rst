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

- `Compare model`_ performances
- Your `first tournament`_
- `Backtest`_ example
- The `run and splitter`_ functions

Miscellaneous
--------------

- `Transform features`_
- Calculate `concordance`_
- Numerai's `CV warning`_  to hold out eras not rows


.. _data: https://github.com/kwgoodman/numerox/blob/master/examples/data.rst
.. _model: https://github.com/kwgoodman/numerox/blob/master/numerox/model.py
.. _prediction: https://github.com/kwgoodman/numerox/blob/master/examples/prediction.rst

.. _compare model: https://github.com/kwgoodman/numerox/blob/master/examples/compare_models.rst
.. _first tournament: https://github.com/kwgoodman/numerox/blob/master/examples/first_tournament.py
.. _backtest: https://github.com/kwgoodman/numerox/blob/master/examples/backtest_example.py
.. _run and splitter: https://github.com/kwgoodman/numerox/blob/master/examples/run.rst

.. _Transform features: https://github.com/kwgoodman/numerox/blob/master/examples/transform.rst
.. _concordance: https://github.com/kwgoodman/numerox/blob/master/examples/concordance_example.py
.. _cv warning: https://github.com/kwgoodman/numerox/blob/master/examples/cv_warning.rst
