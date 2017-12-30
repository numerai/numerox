Comparing model performance
===========================

Let's run multiple models through a simple cross validation on the training
data and then compare the performance of the models. The code for this
example is `here`_.

First perform the cross validation::

    >>> report = nx.Report()
    >>> report['logistic'] = nx.backtest(nx.logistic(), data, verbosity=1)
    >>> report['extratrees'] = nx.backtest(nx.extratrees(), data, verbosity=1)
    >>> report['randomforest'] = nx.backtest(nx.randomforest(), data, verbosity=1)
    >>> report['mlpc'] = nx.backtest(nx.mlpc(), data, verbosity=1)
    >>> report['logisticPCA'] = nx.backtest(nx.logisticPCA(), data, verbosity=1)
    logistic(inverse_l2=0.0001)
          logloss   auc     acc     ystd   stats
    mean  0.692885  0.5165  0.5116  0.0056  region     train
    std   0.000536  0.0281  0.0215  0.0003    eras       120
    min   0.691360  0.4478  0.4540  0.0050  sharpe  0.488866
    max   0.694202  0.5944  0.5636  0.0061  consis  0.691667
    extratrees(depth=3, ntrees=100, seed=0, nfeatures=7)
          logloss   auc     acc     ystd   stats
    mean  0.692948  0.5155  0.5108  0.0044  region     train
    std   0.000453  0.0296  0.0227  0.0003    eras       120
    min   0.691592  0.4322  0.4422  0.0039  sharpe  0.440766
    max   0.694299  0.5986  0.5767  0.0050  consis     0.675
    randomforest(max_features=2, depth=3, ntrees=100, seed=0)
          logloss   auc     acc     ystd   stats
    mean  0.692899  0.5160  0.5114  0.0056  region     train
    std   0.000570  0.0293  0.0218  0.0003    eras       120
    min   0.691133  0.4389  0.4529  0.0051  sharpe  0.435935
    max   0.694459  0.6026  0.5734  0.0061  consis  0.691667
    mlpc(layers=[5, 3], alpha=0.11, activation=tanh, seed=0, learn=0.002)
          logloss   auc     acc     ystd   stats
    mean  0.692867  0.5168  0.5098  0.0093  region     train
    std   0.000958  0.0284  0.0197  0.0022    eras       120
    min   0.689855  0.4516  0.4622  0.0061  sharpe  0.292265
    max   0.695146  0.5952  0.5659  0.0128  consis     0.675
    logisticPCA(nfeatures=10, inverse_l2=0.0001)
          logloss   auc     acc     ystd   stats
    mean  0.692898  0.5159  0.5111  0.0055  region     train
    std   0.000475  0.0255  0.0196  0.0003    eras       120
    min   0.691492  0.4497  0.4590  0.0050  sharpe  0.525184
    max   0.694138  0.5887  0.5653  0.0060  consis  0.708333

Notice how the predictions from the models are highly correlated::

    >>> report.correlation('logistic')
    logistic
       0.9837 logisticPCA
       0.9514 extratrees
       0.9303 randomforest
       0.8392 mlpc

Comparison of model performance::

    >>> report.performance(data, sort_by='logloss')
    train; 120 eras
                  logloss   auc     acc     ystd    sharpe  consis
    model
    mlpc          0.692867  0.5168  0.5098  0.0093  0.2923  0.6750
    logistic      0.692885  0.5165  0.5116  0.0056  0.4889  0.6917
    logisticPCA   0.692898  0.5159  0.5111  0.0055  0.5252  0.7083
    randomforest  0.692899  0.5160  0.5114  0.0056  0.4359  0.6917
    extratrees    0.692948  0.5155  0.5108  0.0044  0.4408  0.6750

Next, let's look at model dominance. For each model calculate what fraction
of models it beats (in terms of logloss) in each era. Then take the mean for
each model across all eras. Repeat for auc and acc. A score of 1 means the
model was the top performer in every era; a score of 0 means the model was the
worst performer in every era::

    >>> report.dominance(data, sort_by='logloss')
                  logloss  auc     acc
    mlpc          0.5771   0.5479  0.4625
    logistic      0.5417   0.5250  0.5312
    logisticPCA   0.5229   0.4646  0.4938
    randomforest  0.5042   0.4938  0.5333
    extratrees    0.3542   0.4688  0.4396

Let's say you have already submitted the predictions of the logistic model.
Which of your other models will that submission prevent from passing
originality::

    >>> print(report.originality(['logistic']))
                   corr     ks  original
    randomforest   True  False     False
    extratrees    False   True     False
    mlpc           True   True      True
    logisticPCA   False  False     False

Typically you would use ``report.originality`` with tournament predictions.
Here the predictions are on the training data.

.. _here: https://github.com/kwgoodman/numerox/blob/master/examples/compare_models.py
