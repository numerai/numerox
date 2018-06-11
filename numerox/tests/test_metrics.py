from nose.tools import assert_raises

from numerox import testing
from numerox.metrics import metrics_per_era
from numerox.metrics import metrics_per_name


def test_metrics_per_era():
    "make sure metrics_per_era runs"
    d = testing.micro_data()
    p = testing.micro_prediction()
    metrics_per_era(d, p, 1)
    metrics_per_era(d, p, 2, join='yhat')
    metrics_per_era(d, p, 3, join='inner')
    assert_raises(ValueError, metrics_per_era, d, p, 4, 'outer')
    with testing.HiddenPrints():
        metrics_per_era(d, p, tournament=5, era_as_str=True)


def test_metrics_per_name():
    "make sure metrics_per_name runs"
    d = testing.micro_data()
    p = testing.micro_prediction()
    metrics_per_name(d, p, 1)
    metrics_per_name(d, p, 2, join='yhat')
    metrics_per_name(d, p, 3, columns=['sharpe'])
    assert_raises(ValueError, metrics_per_name, d, p, 4, 'data', ['wtf'])
