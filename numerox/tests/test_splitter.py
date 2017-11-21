from nose.tools import ok_

from numerox.testing import micro_data
from numerox.splitter import (TournamentSplitter, ValidationSplitter,
                              CheatSplitter, CVSplitter, SplitSplitter)


def test_splitter_overlap():
    "prediction data should not overlap"
    d = micro_data()
    splitters = [TournamentSplitter(d),
                 ValidationSplitter(d),
                 CheatSplitter(d),
                 CVSplitter(d, kfold=2),
                 SplitSplitter(d, fit_fraction=0.5)]
    for splitter in splitters:
        predict_ids = []
        for dfit, dpredict in splitter:
            predict_ids.extend(dpredict.ids.tolist())
        ok_(len(predict_ids) == len(set(predict_ids)), "ids overlap")
