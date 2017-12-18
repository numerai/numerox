Comparing model performance
===========================

Let's run multiple models through a simple cross validation on the training
data and then compare the performance of the models. The code for this
example is `here`_.

First perform the cross validation::

    >>> runner = nx.Runner(run_list, splitter, save_dir, verbosity=1)
    >>> runner.run()
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
    xgboost(n_estimators=5, subsample=0.4, learning_rate=0.1, seed=0, max_depth=5)
          logloss   auc     acc     ystd   stats            
    mean  0.692895  0.5136  0.5093  0.0087  region     train
    std   0.000633  0.0224  0.0170  0.0004    eras       120
    min   0.691090  0.4523  0.4592  0.0078  sharpe  0.398632
    max   0.694656  0.5730  0.5501  0.0095  consis  0.666667
    logisticPCA(nfeatures=10, inverse_l2=0.0001)
          logloss   auc     acc     ystd   stats            
    mean  0.692898  0.5159  0.5110  0.0055  region     train
    std   0.000475  0.0255  0.0196  0.0003    eras       120
    min   0.691492  0.4497  0.4590  0.0050  sharpe  0.525184
    max   0.694138  0.5887  0.5653  0.0060  consis  0.708333

Notice how the predictions from the models are highly correlated::

    >>> report.correlation('logistic')
    logistic
       0.9837 logisticPCA
       0.9514 extratrees
       0.9303 randomforest
       0.6666 xgboost

Comparison of model performance::

    >>> report.performance(data, sort_by='logloss')
    train; 120 eras
                  logloss   auc     acc     ystd    sharpe  consis
    model                                                         
    logistic      0.692885  0.5165  0.5116  0.0056  0.4889  0.6917
    xgboost       0.692895  0.5136  0.5093  0.0087  0.3986  0.6667
    logisticPCA   0.692898  0.5159  0.5110  0.0055  0.5252  0.7083
    randomforest  0.692899  0.5160  0.5114  0.0056  0.4359  0.6917
    extratrees    0.692948  0.5155  0.5108  0.0044  0.4408  0.6750

.. _here: https://github.com/kwgoodman/numerox/blob/master/examples/runner_example.py
