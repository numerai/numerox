from nose.tools import ok_

from numerox.testing import micro_data, play_data
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


def test_splitter_reset():
    "splitter reset should not change results"
    d = micro_data()
    splitters = [TournamentSplitter(d),
                 ValidationSplitter(d),
                 CheatSplitter(d),
                 CVSplitter(d, kfold=2),
                 SplitSplitter(d, fit_fraction=0.5)]
    for splitter in splitters:
        ftups = [[], []]
        ptups = [[], []]
        for i in range(2):
            for dfit, dpredict in splitter:
                ftups[i].append(dfit)
                ptups[i].append(dpredict)
            splitter.reset()
        ok_(ftups[0] == ftups[1], "splitter reset changed fit split")
        ok_(ptups[0] == ptups[1], "splitter reset changed predict split")


def test_cvsplitter_kfold():
    "make sure cvsplitter runs k folds"
    d = play_data()
    for k in (2, 3):
        splitter = CVSplitter(d, kfold=k)
        count = 0
        for dfit, dpredict in splitter:
            count += 1
        ok_(count == k, "CVSplitter iterated through wrong number of folds")
