from nose.tools import assert_raises

from numerox.metrics import metrics_per_era
from numerox.testing import micro_prediction, micro_data


def test_metrics_per_era():
    "make sure calc_metrics runs"
    d = micro_data()
    p = micro_prediction()
    metrics_per_era(d, p)
    metrics_per_era(d, p, 'yhat')
    metrics_per_era(d, p, 'inner')
    assert_raises(ValueError, metrics_per_era, d, p, 'outer')
