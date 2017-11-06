from nose.tools import ok_

from numerox.testing import micro_data
from numerox.splitter import (tournament_splitter, validation_splitter,
                              cheat_splitter, cv_splitter, split_splitter)


def test_splitter_overlap():
    "prediction data should not overlap"
    d = micro_data()
    splitters = [tournament_splitter(d),
                 validation_splitter(d),
                 cheat_splitter(d),
                 cv_splitter(d, kfold=2),
                 split_splitter(d, fit_fraction=0.5)]
    for splitter in splitters:
        predict_ids = []
        for dfit, dpredict in splitter:
            predict_ids.extend(dpredict.ids.tolist())
        ok_(len(predict_ids) == len(set(predict_ids)), "ids overlap")
