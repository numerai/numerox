Comparing model performance
===========================

Let's use the logistic regression model in numerox to run 5-fold cross
validation on the training data::

    >>> model = nx.model.logistic()
    >>> prediction1 = nx.backtest(model, data, verbosity=1)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.692974  0.5226  0.5159  0.0023  |  region   train
    std   0.000224  0.0272  0.0205  0.0002  |  eras     85
    min   0.692360  0.4550  0.4660  0.0020  |  consis   0.7647
    max   0.693589  0.5875  0.5606  0.0027  |  75th     0.6931

Let's take a peek at performance on the validation data::

    >>> prediction2 = nx.production(model, data)
    logistic(inverse_l2=1e-05)
          logloss   auc     acc     ystd
    mean  0.692993  0.5157  0.5115  0.0028  |  region   validation
    std   0.000225  0.0224  0.0172  0.0000  |  eras     12
    min   0.692440  0.4853  0.4886  0.0028  |  consis   0.7500
    max   0.693330  0.5734  0.5555  0.0028  |  75th     0.6931
    >>> prediction2.to_csv('logistic.csv')  # 6 decimal places by default

There is no overlap in ids between prediction1 (train) and prediction2
(tournament) so you can add (concatenate) them if you're into that and let's
go ahead and save the result::

    >>> prediction = prediction1 + prediction2
    >>> prediction.save('logloss_1e-05.pred')  # HDF5

Once you have run and saved several predictions, you can make a report::

    >>> report = nx.report.load_report('/round79', extension='pred')
    >>> report.performance(data['train'], sort_by='logloss')
    logloss   auc     acc     ystd    consis (train; 85 eras)
    0.692455  0.5215  0.5149  0.0219  0.6824        logistic_1e-03
    0.692487  0.5224  0.5159  0.0121  0.7294        logistic_1e-04
    0.692565  0.5236  0.5162  0.0086  0.7294  extratrees_nfeature7
    0.692581  0.5206  0.5143  0.0253  0.6000        logistic_1e-02
    0.692629  0.5240  0.5164  0.0074  0.7294  extratrees_nfeature5
    0.692704  0.5200  0.5140  0.0273  0.5412        logistic_1e-01
    0.692747  0.5232  0.5162  0.0055  0.7647  extratrees_nfeature3
    0.692831  0.5238  0.5163  0.0042  0.7647  extratrees_nfeature2
    0.692974  0.5226  0.5159  0.0023  0.7647        logistic_1e-05

The lowest logloss on the train data was by ``logistic_1e-03``. Let's look at
its per era performance on the validation data::

    >>> report.performance_per_era(data['validation'], 'logistic_1e-03')
    logistic_1e-03
           logloss   auc     acc     ystd
    era86  0.691499  0.5322  0.5296  0.0220
    era87  0.689715  0.5552  0.5371  0.0219
    era88  0.692501  0.5189  0.5167  0.0220
    era89  0.694544  0.4954  0.4916  0.0218
    era90  0.691133  0.5349  0.5230  0.0221
    era91  0.692794  0.5140  0.5061  0.0218
    era92  0.694579  0.4933  0.4906  0.0217
    era93  0.694098  0.4983  0.4954  0.0218
    era94  0.688417  0.5752  0.5591  0.0218
    era95  0.691734  0.5265  0.5224  0.0216
    era96  0.693184  0.5119  0.5092  0.0215
    era97  0.693276  0.5077  0.5089  0.0215
