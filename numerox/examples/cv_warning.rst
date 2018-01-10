Cross validation warning
========================

Numerai warns us: to avoid overfitting hold out a sample of eras not rows
when doing cross validation. That makes sense: what works for one random sample
of stocks tends to work for another sample in the same time period.
Cross-sectional returns can often be explained well by a few linear factors.

Let's use numerox to test the warning for one particular model (logistic
regression with the default regularization). The code for this example is
`here`_.

We will do 100 cross validations that uses eras as Numerai recommends (cve)
and 100 with a traditional cross validation that ignores eras (cv). Here are
the mean results::

    100 runs
                  cve        cv
    logloss  0.692899  0.692813
    auc      0.515948  0.520311
    acc      0.511405  0.514947
    ystd     0.005586  0.005455
    sharpe   0.453231  0.662968
    consis   0.692333  0.783167

Every measure does better (that's the over fit) by ignoring eras (cv). Without
calculating a significance (which I didn't do) the results are not very
meaningful. But casual observation of the first 10 runs, if I remember
correctly, showed that cv won logloss every time.

Not shown in the results but seen when watching the results accumulate, the
logloss of cv is less noisy than that of cve. That makes sense too. In cv
there is very likely to be fit and predict data from every era in every cross
validation run.


.. _here: https://github.com/kwgoodman/numerox/blob/master/numerox/examples/cv_warning.py
