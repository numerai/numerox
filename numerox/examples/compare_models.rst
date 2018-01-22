Comparing model performance
===========================

Let's run multiple models, fitting to the training data and predicting the
validation data. Then we'll compare the performance of the models. The code
for this example is `here`_.

First run the models::

    >>> prediction = nx.production(nx.logistic(), data, verbosity=1)
    >>> prediction += nx.production(nx.extratrees(), data, verbosity=1)
    >>> prediction += nx.production(nx.randomforest(), data, verbosity=1)
    >>> prediction += nx.production(nx.mlpc(), data, verbosity=1)
    >>> prediction += nx.production(nx.logisticPCA(), data, verbosity=1)

which gives the output::

    logistic(inverse_l2=0.0001)
          logloss   auc     acc     ystd   stats
    mean  0.692808  0.5194  0.5142  0.0063  region  validation
    std   0.000375  0.0168  0.0137  0.0001    eras          12
    min   0.691961  0.4903  0.4925  0.0062  sharpe    0.903277
    max   0.693460  0.5553  0.5342  0.0064  consis    0.916667
    extratrees(depth=3, ntrees=100, seed=0, nfeatures=7)
          logloss   auc     acc     ystd   stats
    mean  0.692921  0.5179  0.5132  0.0044  region  validation
    std   0.000272  0.0163  0.0122  0.0000    eras          12
    min   0.692255  0.4915  0.4946  0.0043  sharpe    0.830128
    max   0.693381  0.5569  0.5396  0.0044  consis    0.916667
    randomforest(max_features=2, depth=3, ntrees=100, seed=0)
          logloss   auc     acc     ystd   stats
    mean  0.692874  0.5177  0.5137  0.0056  region  validation
    std   0.000303  0.0139  0.0112  0.0001    eras          12
    min   0.692198  0.4911  0.4983  0.0054  sharpe    0.899559
    max   0.693463  0.5480  0.5380  0.0057  consis    0.916667
    mlpc(layers=[5, 3], alpha=0.11, activation=tanh, seed=0, learn=0.002)
          logloss   auc     acc     ystd   stats
    mean  0.692801  0.5205  0.5124  0.0079  region  validation
    std   0.000466  0.0170  0.0082  0.0001    eras          12
    min   0.691804  0.4915  0.5022  0.0077  sharpe    0.742595
    max   0.693588  0.5553  0.5314  0.0081  consis    0.833333
    logisticPCA(nfeatures=10, inverse_l2=0.0001)
          logloss   auc     acc     ystd   stats
    mean  0.692785  0.5205  0.5140  0.0063  region  validation
    std   0.000361  0.0165  0.0116  0.0001    eras          12
    min   0.691995  0.4912  0.4973  0.0061  sharpe     1.00304
    max   0.693446  0.5546  0.5355  0.0064  consis    0.916667

Notice how the predictions from the models are highly correlated::

    >>> prediction.correlation('logistic')
    logistic
       0.9948 mlpc
       0.9847 logisticPCA
       0.9475 extratrees
       0.9300 randomforest

Also notice that the name of the prediction is by default the name of the
model (you can pick another name).

Comparison of model performance::

    >>> prediction.performance(data, sort_by='logloss')
    validation; 12 eras
                  logloss   auc     acc     ystd    sharpe  consis
    name
    logisticPCA   0.692785  0.5205  0.5140  0.0063  1.0030  0.9167
    mlpc          0.692801  0.5205  0.5124  0.0079  0.7426  0.8333
    logistic      0.692808  0.5194  0.5142  0.0063  0.9033  0.9167
    randomforest  0.692874  0.5177  0.5137  0.0056  0.8996  0.9167
    extratrees    0.692921  0.5179  0.5132  0.0044  0.8301  0.9167

Next, let's look at model dominance. For each model calculate what fraction
of models it beats (in terms of logloss) in each era. Then take the mean for
each model across all eras. Repeat for auc and acc. A score of 1 means the
model was the top performer in every era; a score of 0 means the model was the
worst performer in every era::

    >>> prediction.dominance(data, sort_by='logloss')
                  logloss  auc     acc
    logisticPCA   0.7500   0.6875  0.4792
    logistic      0.6042   0.3542  0.4583
    mlpc          0.5625   0.6667  0.5417
    randomforest  0.3750   0.3542  0.5625
    extratrees    0.2083   0.4375  0.4167

Let's compare dominance of two models::

    >>> prediction[['logistic', 'logisticPCA']].dominance(data['validation'])
             logloss  auc   acc
    logisticPCA  0.6667   0.75  0.5000
    logistic     0.3333   0.25  0.4167

Let's say you have already submitted the predictions of the logistic model.
Which of your other models will that submission prevent from passing
originality::

    >>> print(prediction.originality(['logistic']))
                   corr     ks  original
    extratrees     True   True      True
    randomforest   True   True      True
    mlpc          False   True     False
    logisticPCA   False  False     False

A concordance if less than 0.12 is needed to pass Numerai's test::

    >>> print(prediction.concordance(data))
                     concord
    randomforest    0.029362
    extratrees     0.0341881
    logisticPCA    0.0358237
    mlpc           0.0369206
    logistic       0.0392527

.. _here: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/compare_models.py
