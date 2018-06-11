Numerai upload checks
=====================

If your tournament submission does not pass Numerai's upload checks then
Numerai will reject the submission immediately. You can use Numerox to make
sure the checks will pass before you upload.

First load the dataset::

    >>> data = nx.load_zip('numerai_dataset.zip')

Next make a prediction::

    >>> model = nx.logistic()
    >>> prediction = nx.production(model, data, tournament='bernie')

Or, alternatively, if you do not use Numerox to run your model then you can
create the prediction from 1-dimensional numpy arrays of the Numerai row ids,
``ids``, and the prediction values, ``yhat``::

    >>> prediction = nx.Prediction()
    >>> prediction = prediction.merge_arrays(ids, yhat, 'my_model_name')

Let's run the checks::

    >>> check = prediction.check(data, example_predictions='bernie')
    logistic_bernie
          validation      test      live       all  pass
    corr    0.868204  0.861861  0.869963   0.86325  True
    rcorr   0.868637  0.862757  0.874491  0.864123  True
    min     0.475277  0.476348  0.475243  0.475243  True
    max      0.52378  0.524316  0.518989  0.524316  True
    maz       3.8993   3.92653    3.8589   3.96304  True

All checks passed!

If you pass the tournament number or tournament name to the ``check`` method
then numerox will calculate the example prediction. Alternatively, to run
fast if you wish to check more than one model, you can pass in the example
predictions as a prediction object, which you can generate in one of two ways::

    >>> example_predictions = nx.load_example_predictions('data.zip', 'bernie')

or::

    >>> model = nx.example_predictions()
    >>> example_predictions = nx.production(data, model, 'bernie')
