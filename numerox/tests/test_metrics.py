from nose.tools import assert_raises

from numerox.metrics import calc_metrics
from numerox.testing import micro_prediction, micro_data


def test_calc_metrics():
    "make sure calc_metrics runs"
    d = micro_data()
    p = micro_prediction()
    calc_metrics(d, p)
    calc_metrics(d, p, 'yhat')
    calc_metrics(d, p, 'inner')
    assert_raises(ValueError, calc_metrics, d, p, 'outer')
