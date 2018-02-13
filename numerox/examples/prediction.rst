Prediction
==========

To see the Prediction class in action have a look at this `example`_ and
corresponding `documentation`_.

Let's create some Prediction objects. Yay!

An empty prediction::

    >>> import numerox as nx
    >>> prediction = nx.Prediction()

Running a model returns a prediction::

    >>> prediction = nx.production(nx.logstic(), data)

By default the name of the prediction is the model name::

    >>> prediction.names
    >>> ['logistic']

We can change the default name to, say, 'logret'::

    >>> prediction = nx.production(nx.logstic(), data, name='logret')

Or::

    >>> prediction = nx.Prediction()
    >>> prediction['logret'] = nx.production(nx.logstic(), data)

A prediction object can hold predictions from more than one model::

    >>> prediction = nx.production(nx.logstic(), data)
    >>> prediction += nx.production(nx.extratrees(), data)
    >>> prediction += nx.production(nx.randomforest(), data)

Or::

    >>> prediction = nx.Prediction()
    >>> prediction['rf_d1'] = nx.production(nx.randomforest(depth=1), data)
    >>> prediction['rf_d2'] = nx.production(nx.randomforest(depth=2), data)
    >>> prediction['rf_d3'] = nx.production(nx.randomforest(depth=3), data)

You can save your predictions to a HDF5 file for later use::

    >>> prediction.save('mypredictions.h5')

And then load them::

    >>> prediction = nx.load_prediction('mypredictions.h5')

And you can save one model's predictions to csv for future upload to Numerai::

    >>> prediction['rf3_d3'].to_csv('rf_d3.csv')

It is better to load your predictions from ain HDF5 file (faster, no rounding
errors, can contain predictions from multiple models) but you can load from
an csv file which might be useful when checking a csv file that you submitted
to Numerai::

    >>> prediction2 = nx.load_prediction_csv('rf_d3.csv')

I forget, did I try a depth of 5::

    >>> 'rf_d5' in prediction
    False

If you only want to look at the performance of one run::

    >>> prediction['rf_d3'].performance(data['validation'])

Or compare two models for dominance::

    >>> prediction[['rf_d2', 'rf_d3']].dominance(data['validation'])


.. _example: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/compare_models.py
.. _documentation: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/compare_models.rst

