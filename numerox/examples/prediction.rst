Prediction
==========

Let's take a tour of numerox's Prediction object.

Create a Prediction object
--------------------------

There are three common ways to create a Prediction object.

If you have already ported your model to numerox then running your model
returns a Prediction object. For example::

    >>> import numerox as nx
    >>> p = nx.production(my_model(), data, 'bernie')

where ``p`` is a Prediction object and `data`_ is a Data object.

If you have not yet ported your model to numerox then load your predictions
from a Numerai-style csv file::

    >>> p = nx.load_prediction_csv('my_model.csv')

If your existing code already has both predictions ``yhat`` and Numerai row
``ids`` as numpy arrays::

    >>> p = nx.Prediction()
    >>> p = p.merge_arrays(ids, yhat, 'my_model')

Multiple models
---------------

A Prediction object can contain the predictions of multiple models::

    >>> p = nx.production(my_model_1(), data, 'bernie')
    >>> p += nx.production(my_model_2(), data, 'bernie')

or::

    >>> p = nx.load_prediction_csv('my_model_1.csv')
    >>> p += nx.load_prediction_csv('my_model_2.csv')

or::

    >>> p = nx.Prediction()
    >>> p = p.merge_arrays(ids, yhat, 'my_model_1')
    >>> p += p.merge_arrays(ids, yhat, 'my_model_2')

or::

    >>> p = p1 + p2

or::

    >>> p = p.merge(p2)

Evaluate predictions
--------------------

Let's start by running some models::

    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> p = nx.production(nx.logistic(), data, 'bernie')
    >>> p += nx.production(nx.randomforest(), data, 'bernie')
    >>> p += nx.production(nx.example_predictions(), data, 'bernie')

which gives the output::

    logistic(inverse_l2=0.0001)
           logloss     auc     acc    ystd   stats            
    mean  0.692808  0.5194  0.5142  0.0063   tourn      bernie
    std   0.000375  0.0168  0.0137  0.0001  region  validation
    min   0.691961  0.4903  0.4925  0.0062    eras          12
    max   0.693460  0.5553  0.5342  0.0064  consis        0.75
    Done in 0.05 minutes

    randomforest(max_features=2, depth=3, ntrees=100, seed=0)
           logloss     auc     acc    ystd   stats            
    mean  0.692874  0.5176  0.5138  0.0055   tourn      bernie
    std   0.000303  0.0138  0.0115  0.0001  region  validation
    min   0.692236  0.4902  0.4954  0.0054    eras          12
    max   0.693495  0.5457  0.5342  0.0056  consis    0.666667
    Done in 0.14 minutes
    
    example_predictions()
           logloss     auc     acc    ystd   stats            
    mean  0.692867  0.5150  0.5121  0.0081   tourn      bernie
    std   0.000435  0.0144  0.0115  0.0001  region  validation
    min   0.691768  0.4959  0.4960  0.0080    eras          12
    max   0.693519  0.5508  0.5378  0.0082  consis    0.583333
    Done in 0.26 minutes

We can rerun the performance summaries at any time::

    >>> df_dict = p.summaries(data['validation'], 'bernie')
    logistic_bernie
           logloss     auc     acc    ystd   stats            
    mean  0.692808  0.5194  0.5142  0.0063   tourn      bernie
    std   0.000375  0.0168  0.0137  0.0001  region  validation
    min   0.691961  0.4903  0.4925  0.0062    eras          12
    max   0.693460  0.5553  0.5342  0.0064  consis        0.75
    randomforest_bernie
           logloss     auc     acc    ystd   stats            
    mean  0.692874  0.5176  0.5138  0.0055   tourn      bernie
    std   0.000303  0.0138  0.0115  0.0001  region  validation
    min   0.692236  0.4902  0.4954  0.0054    eras          12
    max   0.693495  0.5457  0.5342  0.0056  consis    0.666667
    example_predictions_bernie
           logloss     auc     acc    ystd   stats            
    mean  0.692867  0.5150  0.5121  0.0081   tourn      bernie
    std   0.000435  0.0144  0.0115  0.0001  region  validation
    min   0.691768  0.4959  0.4960  0.0080    eras          12
    max   0.693519  0.5508  0.5378  0.0082  consis    0.583333

Notice how the predictions from the models are highly correlated::

    >>> p.correlation()
    logistic_bernie
       0.9311 randomforest_bernie
       0.8631 example_predictions_bernie
    randomforest_bernie
       0.9311 logistic_bernie
       0.8935 example_predictions_bernie
    example_predictions_bernie
       0.8935 randomforest_bernie
       0.8631 logistic_bernie

Comparison of model performance sorted by logloss::

    >>> p.performance(data['validation'], 'bernie', sort_by='logloss')
                                 logloss       auc       acc      ystd    sharpe    consis
    name                                                                                  
    logistic_bernie             0.692808  0.519403  0.514200  0.006322  0.510818  0.750000
    example_predictions_bernie  0.692867  0.515008  0.512093  0.008115  0.304800  0.583333
    randomforest_bernie         0.692874  0.517564  0.513843  0.005544  0.414636  0.666667

Even though the models were trained on bernie targets we can evaluate performance
in other tournaments. Let's see how well the predictions perform on Elizabeth targets::

    >>> p.performance(data['validation'], 'elizabeth', sort_by='logloss')
                                 logloss       auc       acc      ystd    sharpe    consis
    name                                                                                  
    example_predictions_bernie  0.692879  0.514126  0.510926  0.008115  0.227803  0.416667
    logistic_bernie             0.692881  0.515328  0.510151  0.006322  0.282277  0.583333
    randomforest_bernie         0.692954  0.512626  0.509476  0.005544  0.129843  0.500000

You can even look at the performance in a single era::

    >>> p.performance(data['era127'], 'bernie', sort_by='logloss')
                                 logloss       auc       acc      ystd  sharpe  consis
    name                                                                              
    example_predictions_bernie  0.692803  0.519303  0.512834  0.008094     NaN     1.0
    randomforest_bernie         0.692944  0.514694  0.504895  0.005543     NaN     1.0
    logistic_bernie             0.693080  0.506166  0.499074  0.006302     NaN     0.0

Next, let's look at model dominance. For each model calculate what fraction
of models it beats (in terms of logloss) in each era. Then take the mean for
each model across all eras. Repeat for auc and acc. A score of 1 means the
model was the top performer in every era; a score of 0 means the model was the
worst performer in every era::

    >>> p.dominance(data['validation'], 'bernie', sort_by='logloss')
                                 logloss       auc       acc
    logistic_bernie             0.666667  0.666667  0.541667
    randomforest_bernie         0.458333  0.416667  0.541667
    example_predictions_bernie  0.375000  0.416667  0.416667

We can also look at performance in every era::

        >>> m = p.metrics_per_era(data['validation'], 'bernie', metrics=['logloss', 'logloss_pass', 'auc'])
        >>> m
                                      name   logloss  logloss_pass       auc
        era                                                                           
        era121             logistic_bernie  0.692785          True  0.520504
        era121         randomforest_bernie  0.692780          True  0.520509
        era121  example_predictions_bernie  0.692964          True  0.509787
        era122             logistic_bernie  0.692467          True  0.537129
        era122         randomforest_bernie  0.692531          True  0.534318
        era122  example_predictions_bernie  0.692620          True  0.522543
        era123             logistic_bernie  0.692980          True  0.512810
        era123         randomforest_bernie  0.693044         False  0.511115
        era123  example_predictions_bernie  0.692703          True  0.521525
        era124             logistic_bernie  0.692617          True  0.527354
        era124         randomforest_bernie  0.692824          True  0.521125
        <snip>

Let's zoom in on logloss::

    >>> m.pivot(columns='name', values='logloss')
    name    example_predictions_bernie  logistic_bernie  randomforest_bernie
    era                                                                     
    era121                    0.692964         0.692785             0.692780
    era122                    0.692620         0.692467             0.692531
    era123                    0.692703         0.692980             0.693044
    era124                    0.693064         0.692617             0.692824
    era125                    0.693169         0.692895             0.692885
    era126                    0.692607         0.692561             0.692824
    era127                    0.692803         0.693080             0.692944
    era128                    0.692923         0.693008             0.693063
    era129                    0.691768         0.691961             0.692236
    era130                    0.693176         0.692914             0.692854
    era131                    0.693094         0.692973             0.693009
    era132                    0.693519         0.693460             0.693495

Instead of evaluting performance per era we can evaluate performance per
tournament::

    >>> m = p.metric_per_tournament(data['validation'], metric='logloss')
    >>> m
                                  bernie  elizabeth    jordan       ken   charles      mean
    name                                                                                   
    logistic_bernie             0.692808   0.692881  0.692825  0.692750  0.692842  0.692821
    example_predictions_bernie  0.692867   0.692879  0.692876  0.692797  0.692893  0.692863
    randomforest_bernie         0.692874   0.692954  0.692884  0.692813  0.692894  0.692884

Note that every model above was trained on bernie targets yet obtains the
lowest logloss on ken targets.

Upload checks
-------------

Do the predictions pass concordance? A concordance of less than 0.12 is needed
to pass Numerai's test (so, yes, they all pass)::
  
    >>> p.concordance(data)
                                  concord
    example_predictions_bernie  0.0394123
    randomforest_bernie         0.0434942
    logistic_bernie             0.0476868

If your tournament submission does not pass Numerai's upload checks then
Numerai will reject the submission immediately. You can use Numerox to make
sure the checks will pass before you upload.

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

Save and load
-------------

You can save your predictions to a HDF5 file for later use::

    >>> p.save('predictions.h5')

And then load them::

    >>> p = nx.load_prediction('predictions.h5')

And you can save one model's predictions to csv for future upload to Numerai::

    >>> p['logistic_bernie'].to_csv('logistic_bernie.csv', tournament='bernie')

It is better to load your predictions from an HDF5 file (faster, no rounding
errors, can contain predictions from multiple models) but you can load from
a csv file which might be useful when checking a csv file that you submitted
to Numerai::

    >>> p = nx.load_prediction_csv('logistic_bernie.csv')

Odds and ends
-------------

I forget, is 'logistic_bernie' in the prediction::

    >>> 'logistic_bernie' in p
    True

If you have a lot of models in youe prediction object and only want to
evaluate, say, two of them::

    >>> p2 = p[['model1', 'model2']]

Some other things you can do::

    >>> p.hash()
    7733620780463466132
    >>> p.shape
    (243222, 3)
    >>> len(p)
    243222
    >>> p.size
    729666
    >>> p2 = p.copy()
    >>> p
    Prediction(243222 rows x 3 names; 0.0000 missing)
    >>> p.names
    ['logistic_bernie', 'randomforest_bernie', 'example_predictions_bernie']

But wait! There's more
----------------------

That's enough to get you started. You can now play around with the prediction
object to discover what else it can do.

.. _data: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/data.rst
