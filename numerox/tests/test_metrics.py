from nose.tools import assert_raises

from numerox import testing
from numerox.metrics import metrics_per_era
from numerox.metrics import metrics_per_name


def test_metrics_per_era():
    "make sure metrics_per_era runs"
    d = testing.micro_data()
    p = testing.micro_prediction()
    metrics_per_era(d, p)
    metrics_per_era(d, p, join='yhat')
    metrics_per_era(d, p, join='inner')
    assert_raises(ValueError, metrics_per_era, d, p, 'outer')
    with testing.HiddenPrints():
        metrics_per_era(d, p, era_as_str=True)


def test_metrics_per_name():
    "make sure metrics_per_name runs"
    d = testing.micro_data()
    p = testing.micro_prediction()
    metrics_per_name(d, p)
    metrics_per_name(d, p, join='yhat')
    metrics_per_name(d, p, columns=['sharpe'])
    assert_raises(ValueError, metrics_per_name, d, p, 'data', ['wtf'])
