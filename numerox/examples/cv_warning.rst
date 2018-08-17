Cross validation warning
========================

To avoid overfitting, Numerai warns us, we should hold out a sample of eras not
rows when doing cross validation. That makes sense. What works for one random
sample of stocks tends to work for another sample in the same time period.
Compared to the low signal-to-noise of predicting future stock returns,
cross-sectional same-period returns have very high correlation.

Let's use numerox to test the warning for two models. One is a simple model,
logistic regression, that is less prone to overfitting. And one is a complex
model, a too deep random forest, that is prone to overfitting. The code for
this example is `here`_.

We will do 100 five-fold cross validations that uses eras as Numerai recommends
(cve) and 100 with a traditional cross validation that ignores eras (cv). When
we do CV that ignores eras we do it in a stratfied way that balances the
targets.

Here are the mean results when using the logistic regression model with the
default regularization::

    >>> model = nx.logistic()
    >>> nx.examples.cv_warning(model, data, 'bernie')
    100 runs
                  cve        cv
    logloss  0.692899  0.692813
    auc      0.515948  0.520307
    acc      0.511405  0.514930
    ystd     0.005586  0.005455
    sharpe   0.185913  0.370593
    consis   0.580417  0.650500

Every measure does better (that's the over fit) by ignoring eras (cv).

Let's repeat the experiment using a more complex model, random forest, that is
better at overfitting::

    >>> model = nx.randomforest(ntree=200, depth=7, max_features=10)
    >>> nx.examples.cv_warning(model, data, 'bernie')
    89 runs
                  cve        cv
    logloss  0.692761  0.692448
    auc      0.516010  0.521837
    acc      0.510956  0.515070
    ystd     0.014978  0.014598
    sharpe   0.169746  0.428923
    consis   0.575375  0.677996

The more complex your model, the bigger the stakes are in heeding Numerai's
warning.

.. _here: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/cv_warning.py
