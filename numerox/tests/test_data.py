import os

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from nose.tools import ok_
from nose.tools import assert_raises

import numerox as nx
from numerox import testing
from numerox.testing import shares_memory, micro_data
from numerox.testing import assert_data_equal as ade

TINY_DATASET_CSV = os.path.join(os.path.dirname(__file__),
                                'tiny_dataset_csv.zip')


def test_data_roundtrip():
    "save/load roundtrip shouldn't change data"

    d = micro_data()
    path = None

    try:

        path = testing.create_tempfile('numerox.h5')

        d.save(path)
        d2 = nx.load_data(path)
        ade(d, d2, "data corrupted during roundtrip")

        d.save(path, compress=True)
        d2 = nx.load_data(path)
        ade(d, d2, "data corrupted during roundtrip")

        d = d['live']
        d.save(path)
        d2 = nx.load_data(path)
        ade(d, d2, "data corrupted during roundtrip")

    finally:

        testing.delete_tempfile(path)


def test_data_indexing():
    "test data indexing"

    d = micro_data()

    msg = 'error indexing data by era'
    ade(d['era1'], micro_data([0]), msg)
    ade(d['era2'], micro_data([1, 2]), msg)
    ade(d['era3'], micro_data([3, 4, 5]), msg)
    ade(d['era4'], micro_data([6]), msg)
    ade(d['eraX'], micro_data([7, 8, 9]), msg)

    msg = 'error indexing data by region'
    ade(d['train'], micro_data([0, 1, 2]), msg)
    ade(d['validation'], micro_data([3, 4, 5, 6]), msg)
    ade(d['test'], micro_data([7, 8]), msg)
    ade(d['live'], micro_data([9]), msg)

    msg = 'error indexing data by array'
    ade(d[d.y['bernie'] == 0], micro_data([0, 2, 4, 6, 8, 9]), msg)
    ade(d[d.era == 'era4'], micro_data([6]), msg)

    assert_raises(IndexError, d.__getitem__, 'era')
    assert_raises(IndexError, d.__getitem__, 'wtf')
    assert_raises(IndexError, d.__getitem__, None)


def test_data_slice():
    "test data slicing"

    d = micro_data()

    msg = 'error slicing data by era'
    ade(d['era1':'era2'], micro_data([0, 1, 2]), msg)
    ade(d['era3':'era3'], micro_data([3, 4, 5]), msg)
    ade(d['eraX'], micro_data([7, 8, 9]), msg)
    ade(d['eraX'], micro_data([7, 8, 9]), msg)
    ade(d[::2], micro_data([0, 3, 4, 5, 7, 8, 9]), msg)
    ade(d[:], d, msg)
    ade(d[:'eraX'], d, msg)
    ade(d['era1':], d, msg)

    assert_raises(IndexError, d.__getitem__, slice('erg1', 'era2'))
    assert_raises(IndexError, d.__getitem__, slice('erg1', 'erg2'))
    assert_raises(IndexError, d.__getitem__, slice('era2', 'era1'))
    assert_raises(IndexError, d.__getitem__, slice('era1', 'era2', -2))
    assert_raises(IndexError, d.__getitem__, slice(1, 2))


def test_data_y_indexing():
    "test data y indexing"

    d = micro_data()

    msg = 'y arrays not equal'
    y1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    assert_array_equal(d.y[1], y1, msg)
    assert_array_equal(d.y['bernie'], y1, msg)

    y2 = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    assert_array_equal(d.y[2], y2, msg)
    assert_array_equal(d.y['elizabeth'], y2, msg)

    y3 = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
    assert_array_equal(d.y[3], y3, msg)
    assert_array_equal(d.y['jordan'], y3, msg)

    y4 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert_array_equal(d.y[4], y4, msg)
    assert_array_equal(d.y['ken'], y4, msg)

    y5 = [0, 0, 1, 0, 0, 0, 1, 1, 0, 1]
    assert_array_equal(d.y[5], y5, msg)
    assert_array_equal(d.y['charles'], y5, msg)

    y = np.vstack([[y1], [y2], [y3], [y4], [y5]]).T
    assert_array_equal(d.y[:], y, msg)

    assert_raises(IndexError, d.y.__getitem__, 0)
    assert_raises(IndexError, d.y.__getitem__, 'era')
    assert_raises(IndexError, d.y.__getitem__, 'wtf')
    assert_raises(IndexError, d.y.__getitem__, None)
    assert_raises(IndexError, d.y.__getitem__, slice(1))


def test_data_loc():
    "test data.loc"
    d = micro_data()
    msg = 'data.loc indexing error'
    ade(d.loc[['index1']], micro_data([1]), msg)
    ade(d.loc[['index4']], micro_data([4]), msg)
    ade(d.loc[['index4', 'index0']], micro_data([4, 0]), msg)
    ade(d.loc[['index4', 'index0', 'index2']], micro_data([4, 0, 2]), msg)


def test_data_xnew():
    "test data.xnew"
    d = nx.testing.micro_data()
    x = d.x.copy()
    x = x[:, -2:]
    d2 = d.xnew(x)
    ok_(not shares_memory(d, d2), "data.xnew should return a copy")
    ok_(d2.xshape[1] == 2, "x should have two columns")
    assert_array_equal(d2.x, x, "data.xnew corrupted the values")
    assert_raises(ValueError, d.xnew, x[:4])


def test_data_pca():
    "test data.pca"
    d = nx.play_data()
    nfactors = (None, 3, 0.5)
    for nfactor in nfactors:
        d2 = d.pca(nfactor=nfactor)
        msg = "data.pca should return a copy"
        ok_(not shares_memory(d, d2), msg)
        if nfactor is None:
            ok_(d.shape == d2.shape, "shape should not change")
        corr = np.corrcoef(d2.x.T)
        corr.flat[::corr.shape[0] + 1] = 0
        corr = np.abs(corr).max()
        ok_(corr < 1e-5, "features are not orthogonal")


def test_data_y_for_tournment():
    "test data.y_for_tournmanent"
    d = nx.load_zip(TINY_DATASET_CSV)
    for i in range(1, 6):
        y = np.zeros(14)
        y[i-1] = 1
        y[i-1+5] = 1
        y[10:] = np.nan
        yt = d.y[i]
        yt2 = d.y[nx.tournament_str(i)]
        assert_array_equal(yt, yt2, "y{} indexing corrupted".format(i))
        assert_array_equal(yt, y, "y{} targets corrupted".format(i))


def test_data_y_df():
    "test data.y_df"
    d = micro_data()
    df = d.y_df
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_data_y_similarity():
    "test data.y_similarity"
    d = micro_data()
    df = d.y_similarity()
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_data_y_sum_hist():
    "test data.y_sum_hist"
    d = micro_data()
    df = d.y_sum_hist()
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_data_balance():
    "test data.balance"

    tournament = 1
    d = micro_data()

    # check balance
    b = d.balance(tournament, train_only=False)
    for era in b.unique_era():
        if era != 'eraX':
            y = b[era].y[tournament]
            n0 = (y == 0).sum()
            n1 = (y == 1).sum()
            ok_(n0 == n1, "y is not balanced")

    # check balance
    b = d.balance(tournament, train_only=True)
    eras = np.unique(b.era[b.region == 'train'])
    for era in eras:
        y = b[era].y[tournament]
        n0 = (y == 0).sum()
        n1 = (y == 1).sum()
        ok_(n0 == n1, "y is not balanced")

    # balance already balanced data (regression test)
    d.balance(tournament).balance(tournament)


def test_data_subsample():
    "test data.subsample"
    d = nx.play_data()
    d2 = d.subsample(0.5)
    ok_(isinstance(d2, nx.Data), 'expecting a Data object')
    ok_(d2.shape[0] < 0.51 * d.shape[0], 'expecting smaller subsample')


def test_data_hash():
    "test data.hash"
    d = micro_data()
    ok_(d.hash() == d.hash(), "data.hash not reproduceable")
    d2 = nx.Data(d.df[::2])
    ok_(d2.hash() == d2.hash(), "data.hash not reproduceable")


def test_empty_data():
    "test empty data"
    d = micro_data()
    d['eraXXX']
    d['eraYYY'].__repr__()
    idx = np.zeros(len(d), dtype=np.bool)
    d0 = d[idx]
    ok_(len(d0) == 0, "empty data should have length 0")
    ok_(d0.size == 0, "empty data should have size 0")
    ok_(d0.shape[0] == 0, "empty data should have d.shape[0] == 0")
    ok_(d0.era.size == 0, "empty data should have d.era.size == 0")
    ok_(d0.region.size == 0, "empty data should have d.region.size == 0")
    ok_(d0.x.size == 0, "empty data should have d.x.size == 0")
    ok_(d0.y[:].size == 0, "empty data should have d.y.size == 0")
    d2 = d['era0'] + d[idx]
    ok_(len(d2) == 0, "empty data should have length 0")


def test_data_y_to_nan():
    "test data_y_to_nan"
    d = micro_data()
    ok_(not np.isfinite(d.y_to_nan().y[:]).any(), "not all y's are nan")


def test_data_methods():
    "test data methods"
    d = micro_data()
    ok_(len(d) == 10, "wrong length")
    ok_(d.size == 100, "wrong size")
    ok_(d.shape == (10, 10), "wrong shape")
    ok_(d == d, "not equal")


def test_data_copies():
    "data properties should be copies or views"

    d = micro_data()

    ok_(shares_memory(d, d), "looks like shares_memory failed")

    # copies
    ok_(not shares_memory(d, d.copy()), "should be a copy")
    ok_(not shares_memory(d, d.era), "d.era should be a copy")
    ok_(not shares_memory(d, d.region), "d.region should be a copy")
    ok_(not shares_memory(d, d.ids), "d.ids should be a copy")

    # views
    ok_(shares_memory(d, d.era_float), "d.era_float should be a view")
    ok_(shares_memory(d, d.region_float), "d.region_float should be a view")
    ok_(shares_memory(d, d.x), "d.x should be a view")
    ok_(shares_memory(d, d.y[:]), "d.y[:] should be a view")


def test_data_properties():
    "data properties should not be corrupted"

    d = micro_data()

    ok_((d.ids == d.df.index).all(), "ids is corrupted")
    ok_((d.era_float == d.df.era).all(), "era is corrupted")
    ok_((d.region_float == d.df.region).all(), "region is corrupted")

    idx = ~np.isnan(d.y[:])
    y = d.df[['bernie', 'elizabeth', 'jordan', 'ken', 'charles']].values
    ok_((d.y[:][idx] == y[idx]).all(), "y is corrupted")

    x = d.x
    for i, name in enumerate(d.column_list(x_only=True)):
        ok_((x[:, i] == d.df[name]).all(), "%s is corrupted" % name)


def test_data_era_isnotin():
    "test data.era_isnotin"
    d = micro_data()
    eras = ['era3', 'eraX']
    d0 = d.era_isnotin(eras)
    d1 = d.era_isin(eras)
    d01 = nx.concat_data([d0, d1])
    d01 = d01.loc[d.ids]
    ade(d01, d, "all rows not selected")


def test_data_era_iter():
    "test data.era_iter"
    d = micro_data()
    for as_str in (True, False):
        era2 = []
        for era, idx in d.era_iter(as_str=as_str):
            era2.append(era)
            n = np.unique(d[idx].era).size
            ok_(n == 1, "expecting a single era")
        era = d.unique_era(as_str=as_str).tolist()
        era.sort()
        era2.sort()
        ok_(era2 == era, "era difference found")


def test_data_region_iter():
    "test data.region_iter"
    d = micro_data()
    for as_str in (True, False):
        region2 = []
        for region, idx in d.region_iter(as_str=as_str):
            region2.append(region)
            n = np.unique(d[idx].region).size
            ok_(n == 1, "expecting a single region")
        region = d.unique_region(as_str=as_str).tolist()
        region.sort()
        region2.sort()
        ok_(region2 == region, "region difference found")


def test_data_repr():
    "make sure data__repr__() runs"
    d = micro_data()
    d.__repr__()


# ---------------------------------------------------------------------------
# data functions

def test_concat_data():
    "test concat_data"
    d = nx.testing.micro_data()
    d1 = nx.testing.micro_data(slice(0, 5))
    d2 = nx.testing.micro_data(slice(5, None))
    d12 = nx.concat_data([d1, d2])
    ade(d12, d, "concat_data corrupted adta")
    assert_raises(IndexError, nx.concat_data, [d, d])


def test_load_zip():
    "test nx.load_zip"
    for i in (0, 1):
        if i == 0:
            d = nx.load_zip(TINY_DATASET_CSV)
        else:
            with testing.HiddenPrints():
                d = nx.load_zip(TINY_DATASET_CSV, verbose=True)
        ok_(len(d) == 14, "wrong number of rows")
        ok_(d.shape == (14, 57), 'data has wrong shape')
        ok_(d.x.shape == (14, 50), 'x has wrong shape')
        ok_(d.df.iloc[2, 3] == 0.34143, 'wrong feature value')


def test_compare_data():
    "test compare_data"
    d = nx.testing.micro_data()
    df = nx.compare_data(d, d)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
