import tempfile

import numpy as np
from numpy.testing import assert_array_equal

from nose.tools import ok_
from nose.tools import assert_raises

import numerox as nx
from numerox.testing import shares_memory, micro_data
from numerox.testing import assert_data_equal as ade


def test_data_roundtrip():
    "save/load roundtrip shouldn't change data"
    d = micro_data()
    with tempfile.NamedTemporaryFile() as temp:

        d.save(temp.name)
        d2 = nx.load_data(temp.name)
        ade(d, d2, "data corrupted during roundtrip")

        d.save(temp.name, compress=True)
        d2 = nx.load_data(temp.name)
        ade(d, d2, "data corrupted during roundtrip")

        d = d['live']
        d.save(temp.name)
        d2 = nx.load_data(temp.name)
        ade(d, d2, "data corrupted during roundtrip")


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
    ade(d[d.y == 0], micro_data([0, 2, 4, 6, 8]), msg)
    ade(d[d.era == 'era4'], micro_data([6]), msg)


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


def test_data_balance():
    "test data.balance"
    d = micro_data()
    b = d.balance(train_only=False)
    for era in b.unique_era():
        if era != 'eraX':
            y = b[era].y
            n0 = (y == 0).sum()
            n1 = (y == 1).sum()
            ok_(n0 == n1, "y is not balanced")
    b = d.balance(train_only=True)
    eras = np.unique(b.era[b.region == 'train'])
    for era in eras:
        y = b[era].y
        n0 = (y == 0).sum()
        n1 = (y == 1).sum()
        ok_(n0 == n1, "y is not balanced")


def test_data_hash():
    "test data.hash"
    d = nx.play_data()
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
    ok_(d0.y.size == 0, "empty data should have d.y.size == 0")
    d2 = d['era0'] + d[idx]
    ok_(len(d2) == 0, "empty data should have length 0")


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
    ok_(shares_memory(d, d.y), "d.y should be a view")


def test_data_properties():
    "data properties should not be corrupted"

    d = micro_data()

    ok_((d.ids == d.df.index).all(), "ids is corrupted")
    ok_((d.era_float == d.df.era).all(), "era is corrupted")
    ok_((d.region_float == d.df.region).all(), "region is corrupted")

    idx = ~np.isnan(d.df.y)
    ok_((d.y[idx] == d.df.y[idx]).all(), "y is corrupted")

    x = d.x
    for i, name in enumerate(d._x_names()):
        ok_((x[:, i] == d.df[name]).all(), "%s is corrupted" % name)


def test_data_repr():
    "make sure data__repr__() runs"
    d = micro_data()
    d.__repr__()
