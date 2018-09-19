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

Or you can use one of numerox's builtin models::

    >>> p = nx.production(nx.logistic(), data, 'bernie')

If your existing code already has both predictions ``yhat`` and Numerai row
``ids`` as numpy arrays::

    >>> p = nx.Prediction()
    >>> p = p.merge_arrays(ids, yhat, 'my_model', 'bernie')

Multiple models
---------------

A Prediction object can contain the predictions of multiple models::

    >>> models = [nx.logistic(), nx.randomforest()]
    >>> p = nx.backtest(models, data)

or::

    >>> p = nx.production(my_model_1(), data, 'bernie')
    >>> p += nx.production(my_model_2(), data, 'bernie')

or::

    >>> p = nx.load_prediction_csv('my_model_1.csv')
    >>> p += nx.load_prediction_csv('my_model_2.csv')

or::

    >>> p = nx.Prediction()
    >>> p = p.merge_arrays(ids, yhat, 'my_model_1', 'charles')
    >>> p += p.merge_arrays(ids, yhat, 'my_model_2', 'charles')

or::

    >>> p = p1 + p2

or::

    >>> p = p.merge(p2)

The default is to run a model across all five tournaments)::

    >>> p = nx.production(nx.logistic(), data)


Evaluate predictions
--------------------

Let's start by running some models through all five tournaments::

    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> models = [nx.logistic(), nx.randomforest(), nx.example_predictions()]
    >>> p = nx.production(models, data, verbosity=0)

The (model) names and tournaments contained in the prediction object::

    >>> p
                        bernie elizabeth jordan ken charles
    logistic                 x         x      x   x       x
    randomforest             x         x      x   x       x
    example_predictions      x         x      x   x       x

The mean performance of each model averaged over the five tournaments::

    >>> p.performance_mean(data['validation'], mean_of='name', sort_by='consis')
                         N   logloss       auc       acc      ystd    sharpe    consis
    name
    logistic             5  0.692813  0.520050  0.513910  0.005877  0.548698  0.750000
    randomforest         5  0.692873  0.518583  0.513771  0.005154  0.474319  0.716667
    example_predictions  5  0.692849  0.515931  0.511935  0.007615  0.406432  0.666667

The mean performance in each tournament averaged over the five (model) names::

    >>> p.performance_mean(data['validation'], mean_of='tournament', sort_by='consis')
                N   logloss       auc       acc      ystd    sharpe    consis
    tournament
    charles     3  0.692830  0.521548  0.514893  0.004850  0.719120  0.833333
    ken         3  0.692783  0.520875  0.515404  0.006260  0.661760  0.805556
    bernie      3  0.692848  0.517438  0.513737  0.006672  0.402646  0.694444
    jordan      3  0.692880  0.515928  0.511663  0.006618  0.339453  0.638889
    elizabeth   3  0.692884  0.515149  0.510329  0.006676  0.259435  0.583333

We can look at the detailed performance of each of the 15 runs in the
prediction object::

    >>> p.summaries(data['validation'])
    logistic, bernie
           logloss     auc     acc    ystd   stats
    mean  0.692808  0.5194  0.5142  0.0063   tourn      bernie
    std   0.000375  0.0168  0.0137  0.0001  region  validation
    min   0.691961  0.4903  0.4925  0.0062    eras          12
    max   0.693460  0.5553  0.5342  0.0064  consis        0.75
    logistic, elizabeth
           logloss     auc     acc    ystd   stats
    mean  0.692859  0.5168  0.5101  0.0061   tourn   elizabeth
    std   0.000410  0.0188  0.0137  0.0001  region  validation
    min   0.691911  0.4909  0.4947  0.0060    eras          12
    max   0.693434  0.5593  0.5370  0.0063  consis    0.666667
    <snip>

Or we can look in even more detail by looking at performance in every era::

    >>> p[:, 'bernie'].metric_per_era(data['validation'], metric='logloss')
            (example_predictions, 1)  (logistic, 1)  (randomforest, 1)
    era
    era121                  0.692964       0.692785           0.692743
    era122                  0.692620       0.692467           0.692580
    era123                  0.692703       0.692980           0.693021
    era124                  0.693064       0.692617           0.692869
    era125                  0.693169       0.692895           0.692909
    era126                  0.692607       0.692561           0.692816
    era127                  0.692803       0.693080           0.692931
    era128                  0.692923       0.693008           0.693027
    era129                  0.691768       0.691961           0.692233
    era130                  0.693176       0.692914           0.692813
    era131                  0.693094       0.692973           0.693027
    era132                  0.693519       0.693460           0.693438

or::

    >>> p.metrics_per_era(data['validation'])
                           name tournament   logloss       auc       acc      ystd
    era
    era121             logistic     bernie  0.692785  0.520504  0.520613  0.006376
    era121             logistic  elizabeth  0.692895  0.514934  0.505814  0.006209
    era121             logistic     jordan  0.692871  0.517478  0.512685  0.006324
    era121             logistic        ken  0.692824  0.519075  0.519820  0.005995
    era121             logistic    charles  0.692778  0.526620  0.521406  0.004794
    era121         randomforest     bernie  0.692712  0.522860  0.523520  0.005608
    <snip>

or::

    >>> p['logistic', 'bernie'].metrics_per_era(data['validation'])
                name tournament   logloss       auc       acc      ystd
    era
    era121  logistic     bernie  0.692785  0.520504  0.520613  0.006376
    era122  logistic     bernie  0.692467  0.537129  0.534193  0.006298
    era123  logistic     bernie  0.692980  0.512810  0.507495  0.006316
    era124  logistic     bernie  0.692617  0.527354  0.525091  0.006286
    era125  logistic     bernie  0.692895  0.517678  0.517215  0.006387
    era126  logistic     bernie  0.692561  0.531445  0.519849  0.006438
    era127  logistic     bernie  0.693080  0.506166  0.499074  0.006302
    era128  logistic     bernie  0.693008  0.509709  0.505609  0.006390
    era129  logistic     bernie  0.691961  0.555262  0.532180  0.006443
    era130  logistic     bernie  0.692914  0.515733  0.519341  0.006251
    era131  logistic     bernie  0.692973  0.508727  0.492481  0.006164
    era132  logistic     bernie  0.693460  0.490316  0.497259  0.006216

We can also look in less detail::

    >>> df = p.performance(data['validation'], sort_by='consis')
    >>> print(df.to_string(index=False))
    name tournament   logloss       auc       acc      ystd    sharpe    consis

               logistic        ken  0.692751  0.522883  0.516185  0.005941  0.706879  0.833333
           randomforest        ken  0.692808  0.521669  0.515534  0.005184  0.702168  0.833333
    example_predictions    charles  0.692815  0.518958  0.511656  0.005790  0.713454  0.833333
               logistic    charles  0.692821  0.522683  0.516508  0.004713  0.692814  0.833333
           randomforest    charles  0.692855  0.523003  0.516517  0.004048  0.751093  0.833333
    example_predictions        ken  0.692789  0.518074  0.514492  0.007655  0.576234  0.750000
               logistic     bernie  0.692808  0.519403  0.514200  0.006322  0.510818  0.750000
           randomforest     bernie  0.692868  0.517903  0.514917  0.005578  0.392321  0.750000
               logistic     jordan  0.692826  0.518525  0.512537  0.006284  0.488683  0.666667
               logistic  elizabeth  0.692859  0.516755  0.510123  0.006124  0.344298  0.666667
           randomforest     jordan  0.692891  0.516556  0.512597  0.005590  0.360619  0.666667
    example_predictions  elizabeth  0.692853  0.514909  0.511578  0.008534  0.268613  0.583333
    example_predictions     bernie  0.692867  0.515008  0.512093  0.008115  0.304800  0.583333
    example_predictions     jordan  0.692922  0.512705  0.509855  0.007979  0.169058  0.583333
           randomforest  elizabeth  0.692941  0.513783  0.509287  0.005368  0.165394  0.500000

or::

    >>> df = p[:, 'bernie'].performance(data['validation'], sort_by='consis')
    >>> print(df.to_string(index=False))
    name tournament   logloss       auc       acc      ystd    sharpe    consis

               logistic     bernie  0.692808  0.519403  0.514200  0.006322  0.510818  0.750000
           randomforest     bernie  0.692868  0.517903  0.514917  0.005578  0.392321  0.750000
    example_predictions     bernie  0.692867  0.515008  0.512093  0.008115  0.304800  0.583333

I won't give an example but you can also check the correlation between the
predictions with ``p.correlation()`` and ``p.y_df.corr()``.

Next, let's look at model dominance. For each model calculate what fraction of
models it beats (in terms of logloss) in each era. Then take the mean for each
model across all eras. Repeat for auc and acc. A score of 1 means the model was
the top performer in every era; a score of 0 means the model was the worst
performer in every era. To keep the report short let's only look at 'bernie'::

    >> p[:, 'bernie'].dominance(data['validation'])
                                             name tournament   logloss       auc       acc
    (logistic, 1)                        logistic     bernie  0.708333  0.666667  0.541667
    (randomforest, 1)                randomforest     bernie  0.416667  0.458333  0.625000
    (example_predictions, 1)  example_predictions     bernie  0.375000  0.375000  0.333333

So in about 71% of the eras the logistic model had the lowest logloss.

Indexing
--------

We start with a prediction object, ``p``, that contains::

    >>> p
                        bernie elizabeth jordan ken charles
    logistic                 x         x      x   x       x
    randomforest             x         x      x   x       x
    example_predictions      x         x      x   x       x

You can index by (model) name::

    >>> p['logistic']
             bernie elizabeth jordan ken charles
    logistic      x         x      x   x       x

You can index by tournament::

    >>> p[:, 'ken']
                        bernie elizabeth jordan ken charles
    logistic                                      x
    randomforest                                  x
    example_predictions                           x

You can index by name and tournament::

    >>> p['randomforest', 'charles']
                 bernie elizabeth jordan ken charles
    randomforest                                   x

You can index by (name, tournament) pairs::

    >>> p[[('randomforest', 'charles'), ('logistic', 'jordan')]]
                 bernie elizabeth jordan ken charles
    randomforest                                   x
    logistic                           x

Dropping predictions
--------------------

We start with a prediction object, ``p``, that contains::

    >>> p
                        bernie elizabeth jordan ken charles
    logistic                 x         x      x   x       x
    randomforest             x         x      x   x       x
    example_predictions      x         x      x   x       x

Let's remove the random forest model::

    >>> p.drop_name('randomforest')
                        bernie elizabeth jordan ken charles
    logistic                 x         x      x   x       x
    example_predictions      x         x      x   x       x

Remove the 'ken' tournament::

    >>> p.drop_tournament('ken')
                        bernie elizabeth jordan ken charles
    logistic                 x         x      x           x
    randomforest             x         x      x           x
    example_predictions      x         x      x           x

Remove the logistic, bernie pair::

    >>> p.drop_pair(('logistic', 'bernie'))
                        bernie elizabeth jordan ken charles
    logistic                           x      x   x       x
    randomforest             x         x      x   x       x
    example_predictions      x         x      x   x       x

All three drop functions work with lists as well. For example::

    >>> p.drop_tournament([2, 'ken'])
                        bernie elizabeth jordan ken charles
    logistic                 x                x           x
    randomforest             x                x           x
    example_predictions      x                x           x

Upload checks
-------------

Do the predictions pass concordance? A concordance of less than 0.12 is needed
to pass Numerai's test (so, yes, they all pass)::

    >>> p['logistic'].concordance(data)
                       name tournament    concord
    (logistic, 5)  logistic    charles  0.0398208
    (logistic, 2)  logistic  elizabeth   0.041147
    (logistic, 3)  logistic     jordan   0.042649
    (logistic, 1)  logistic     bernie  0.0430744
    (logistic, 4)  logistic        ken  0.0448813

If your tournament submission does not pass Numerai's upload checks then
Numerai will reject the submission immediately. You can use Numerox to make
sure the checks will pass before you upload.

Let's run the checks::

    >>> p.check(data)
    logistic, bernie
          validation      test      live       all  pass
    corr    0.868204  0.861861  0.868509  0.863216  True
    rcorr   0.868637  0.862757  0.870403  0.864034  True
    min     0.475277  0.476348  0.481861  0.475277  True
    max      0.52378  0.524316  0.522606  0.524316  True
    maz       3.8993   3.92653   3.53621    3.9575  True
    logistic, elizabeth
          validation      test      live       all  pass
    corr    0.830666  0.819013  0.827738  0.821461  True
    rcorr   0.830695  0.819362   0.82823  0.821722  True
    min     0.474478  0.476066  0.481326  0.474478  True
    max     0.522743  0.523472  0.522443  0.523472  True
    maz      4.06343   4.01284   3.70983   4.12892  True
    <snip>

All checks passed!

Save and load
-------------

You can save your predictions to a HDF5 file for later use::

    >>> p.save('predictions.h5')

And then load them::

    >>> p = nx.load_prediction('predictions.h5')

And you can save one model's predictions to csv for future upload to Numerai::

    >>> p['logistic', 'bernie'].to_csv('logistic_bernie.csv')

It is better to load your predictions from an HDF5 file (faster, no rounding
errors, can contain predictions from multiple models) but you can load from
a csv file which might be useful when checking a csv file that you submitted
to Numerai::

    >>> p = nx.load_prediction_csv('logistic_bernie.csv')

Odds and ends
-------------

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
    >>> p.names()
    >>> ['logistic', 'randomforest', 'example_predictions']
    >>> p.tournaments()
    ['bernie', 'elizabeth', 'jordan', 'ken', 'charles']
    >>> p.tournaments(as_str=False)
    [1, 2, 3, 4, 5]
    >>> p.pairs()
    [('logistic', 'bernie'),
     ('logistic', 'elizabeth'),
     ('logistic', 'jordan'),
     ('logistic', 'ken'),
     ('logistic', 'charles'),
     ('randomforest', 'bernie'),
     ('randomforest', 'elizabeth'),
     ('randomforest', 'jordan'),
     ('randomforest', 'ken'),
     ('randomforest', 'charles'),
     ('example_predictions', 'bernie'),
     ('example_predictions', 'elizabeth'),
     ('example_predictions', 'jordan'),
     ('example_predictions', 'ken'),
     ('example_predictions', 'charles')]

But wait! There's more
----------------------

That's enough to get you started. You can now play around with the prediction
object to discover what else it can do.

.. _data: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/data.rst
