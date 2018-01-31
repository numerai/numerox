import zipfile

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

TRAIN_FILE = 'numerai_training_data.csv'
TOURNAMENT_FILE = 'numerai_tournament_data.csv'
HDF_DATA_KEY = 'numerox_data'

ERA_INT_TO_STR = {}
ERA_STR_TO_INT = {}
ERA_STR_TO_FLOAT = {}
for i in range(150):
    name = 'era' + str(i)
    ERA_INT_TO_STR[i] = name
    ERA_STR_TO_INT[name] = i
    ERA_STR_TO_FLOAT[name] = float(i)
ERA_INT_TO_STR[999] = 'eraX'
ERA_STR_TO_INT['eraX'] = 999
ERA_STR_TO_FLOAT['eraX'] = 999.0

TOURNAMENT_REGIONS = ['validation', 'test', 'live']
REGION_INT_TO_STR = {0: 'train', 1: 'validation', 2: 'test', 3: 'live'}
REGION_STR_TO_INT = {'train': 0, 'validation': 1, 'test': 2, 'live': 3}
REGION_STR_TO_FLOAT = {'train': 0., 'validation': 1., 'test': 2., 'live': 3.}


class Data(object):

    def __init__(self, df):
        self.df = df

    # ids -------------------------------------------------------------------

    @property
    def ids(self):
        "Copy of ids as a numpy str array"
        return self.df.index.values.astype('str')

    # era -------------------------------------------------------------------

    @property
    def era(self):
        "Copy of era as a 1d numpy str array"
        series = self.df['era'].map(ERA_INT_TO_STR)
        return series.values.astype(str)

    @property
    def era_float(self):
        "View of era as a 1d numpy float array"
        return self.df['era'].values

    def unique_era(self, as_str=True):
        "Array of unique eras as strings (default) or floats"
        unique_era = self.df.era.unique()
        if as_str:
            unique_era = np.array(self.eras_int2str(unique_era))
        return unique_era

    def era_iter(self, as_str=True):
        "Iterator that yields era and bool index that gives rows of era"
        eras = self.unique_era(as_str=False)
        for era in eras:
            index = self.era_float == era
            if as_str:
                era = ERA_INT_TO_STR[era]
            yield era, index

    def era_isin(self, eras):
        "Copy of data containing only eras in the iterable `eras`"
        eras = self.eras_str2int(eras)
        idx = self.df.era.isin(eras)
        return self[idx]

    def era_isnotin(self, eras):
        "Copy of data containing eras that are not the iterable `eras`"
        eras = self.eras_str2int(eras)
        idx = self.df.era.isin(eras)
        return self[~idx]

    def eras_str2int(self, eras):
        "List with eras names (str) converted to int"
        e = []
        for era in eras:
            if era in ERA_STR_TO_INT:
                e.append(ERA_STR_TO_INT[era])
            else:
                e.append(era)
        return e

    def eras_int2str(self, eras):
        "List with eras numbers converted to eras names (str)"
        e = []
        for era in eras:
            if era in ERA_INT_TO_STR:
                e.append(ERA_INT_TO_STR[era])
            else:
                e.append(era)
        return e

    # region ----------------------------------------------------------------

    @property
    def region(self):
        "Copy of region as a 1d numpy str array"
        series = self.df['region'].map(REGION_INT_TO_STR)
        return series.values.astype(str)

    @property
    def region_float(self):
        "View of region as a 1d numpy float array"
        return self.df['region'].values

    def unique_region(self, as_str=True):
        "Array of unique regions as strings (default) or floats"
        unique_region = self.df.region.unique()
        if as_str:
            unique_region = np.array(self.regions_int2str(unique_region))
        return unique_region

    def region_iter(self, as_str=True):
        "Iterator that yields region and bool index that gives rows of region"
        regions = self.unique_region(as_str=False)
        for region in regions:
            index = self.region_float == region
            if as_str:
                region = REGION_INT_TO_STR[region]
            yield region, index

    def region_isin(self, regions):
        "Copy of data containing only regions in the iterable `regions`"
        regions = self.regions_str2int(regions)
        idx = self.df.region.isin(regions)
        return self[idx]

    def region_isnotin(self, regions):
        "Copy of data containing regions that are not the iterable `regions`"
        regions = self.regions_str2int(regions)
        idx = self.df.region.isin(regions)
        return self[~idx]

    def regions_str2int(self, regions):
        "List with regions names (str) converted to int"
        r = []
        for region in regions:
            if region in REGION_STR_TO_INT:
                r.append(REGION_STR_TO_INT[region])
            else:
                r.append(region)
        return r

    def regions_int2str(self, regions):
        "List with regions numbers converted to region names (str)"
        r = []
        for region in regions:
            if region in REGION_INT_TO_STR:
                r.append(REGION_INT_TO_STR[region])
            else:
                r.append(region)
        return r

    # x ---------------------------------------------------------------------

    @property
    def x(self):
        "View of features, x, as a numpy float array"
        return self.df.iloc[:, 2:-1].values

    def xnew(self, x_array):
        "Copy of data but with data.x=`x_array`; must have same number of rows"
        if x_array.shape[0] != len(self):
            msg = "`x_array` must have the same number of rows as data"
            raise ValueError(msg)
        shape = (x_array.shape[0], x_array.shape[1] + 3)
        cols = ['x'+str(i) for i in range(x_array.shape[1])]
        cols = ['era', 'region'] + cols + ['y']
        df = pd.DataFrame(data=np.empty(shape, dtype=np.float64),
                          index=self.df.index.copy(deep=True),
                          columns=cols)
        df['era'] = self.df['era'].values.copy()
        df['region'] = self.df['region'].values.copy()
        df.values[:, 2:-1] = x_array
        df['y'] = self.df['y'].values.copy()
        return Data(df)

    @property
    def xshape(self):
        "Shape (nrows, ncols) of x; faster than data.x.shape"
        rows = self.df.shape[0]
        cols = len(self.column_list(x_only=True))
        return (rows, cols)

    # y ---------------------------------------------------------------------

    @property
    def y(self):
        "View of y as a 1d numpy float array"
        return self.df['y'].values

    def y_to_nan(self):
        "Copy of data with y values set to NaN"
        data = self.copy()
        data.df = data.df.assign(y=np.nan)
        return data

    # transforms ----------------------------------------------------------

    def pca(self, nfactor=None, data_fit=None):
        """
        Tranform the features (x) using Principal component analysis (PCA).

        Parameters
        ----------
        nfactor : {int, float, None}, optional
            The number of orthogonal features to keep in the transform. By
            default (None) all components are kept. If `nfactor` is less than
            1 then `nfactor` represents the number of factors such that at
            least `nfactor` of the variance is explain. If `nfactor` is
            greater than 1 then it represents the number fo factors to keep.
        data_fit : {Data, None}, optional
            The data used to fit the PCA. By default (None) all data is used.

        Returns
        -------
        data : Data
            A copy of the data object with the requested number of orthogonal
            features.
        """
        if data_fit is None:
            data_fit = self
        if nfactor is None:
            nfactor = self.xshape[1]
        pca = PCA(n_components=nfactor)
        pca.fit(data_fit.x)
        x = pca.transform(self.x)
        data = self.xnew(x)
        return data

    def balance(self, train_only=True, seed=0):
        """
        Copy of data where specified eras have mean y of 0.5.

        Parameters
        ----------
        train_only : {True, False}, optional
            By default (True) only train eras are y balanced. No matter what
            the setting of `train_only` any era that contains a y that is NaN
            is not balanced.
        seed : int, optional
            Seed used by random number generator that selects which rows to
            keep. Default is 0.

        Returns
        -------
        data : Data
            A copy of data where specified eras have mean y of 0.5.
        """
        # This function is not written in a straightforward manner.
        # A few speed optimizations have been made.
        data = self
        if train_only:
            f = REGION_STR_TO_FLOAT['train']
            eras = np.unique(data.era_float[data.region_float == f]).tolist()
        else:
            eras = data.unique_era(as_str=False).tolist()
        era = data.era_float
        y = data.y
        index = np.arange(y.size)
        remove = []
        rs = np.random.RandomState(seed)
        for e in eras:
            idx = era == e
            yi = y[idx]
            indexi = index[idx]
            n1 = yi.sum()
            if np.isnan(n1):
                continue
            n1 = int(n1)
            n0 = yi.size - n1
            if n0 == n1:
                pass
            elif n0 > n1:
                ix = indexi[yi == 0]
                ix = rs.choice(ix, size=n0-n1, replace=False)
                remove.append(ix)
            elif n0 < n1:
                ix = indexi[yi == 1]
                ix = rs.choice(ix, size=n1-n0, replace=False)
                remove.append(ix)
            else:
                msg = "balance should not reach this line"  # pragma: no cover
                raise RuntimeError(msg)  # pragma: no cover
            idx = ~idx
            era = era[idx]
            y = y[idx]
            index = index[idx]
        if len(remove) == 0:
            data = data.copy()
        else:
            keep = set(range(data.shape[0])) - set(np.concatenate(remove))
            keep = list(keep)
            df = data.df.take(keep)
            data = Data(df)
        return data

    def subsample(self, fraction, balance=True, seed=0):
        """
        Randomly sample `fraction` of each era's rows.

        data.y is optionally balanced. The default is to balance y. Balancing
        is achieved by removing rows so the number of rows will likely be
        less than expected using `fraction` if `balance` is True.
        """
        rs = np.random.RandomState(seed)
        data_index = np.arange(len(self))
        era = self.era_float
        eras = self.unique_era(as_str=False)
        index = []
        for e in eras:
            idx = data_index[era == e]
            n = int(fraction * idx.size)
            idx = rs.choice(idx, n, replace=False)
            index.append(idx)
        index = np.concatenate(index)
        df = self.df.take(index)
        data = Data(df)
        if balance:
            data = data.balance(train_only=False, seed=seed)
        return data

    # misc ------------------------------------------------------------------

    def hash(self):
        """
        Hash of data object.

        The hash is unlikely to be the same across different computers (OS,
        Python version, etc). But should be the same for the same dataset on
        the same system.
        """
        b = self.df.values.tobytes(order='A')
        h = hash(b)
        return h

    def copy(self):
        "Copy of data"
        # df.copy(deep=True) doesn't copy index. So:
        df = self.df
        df = pd.DataFrame(df.values.copy(),
                          df.index.copy(deep=True),
                          df.columns.copy())
        return Data(df)

    def save(self, path_or_buf, compress=False):
        "Save data as an hdf archive"
        if compress:
            self.df.to_hdf(path_or_buf, HDF_DATA_KEY,
                           complib='zlib', complevel=4)
        else:
            self.df.to_hdf(path_or_buf, HDF_DATA_KEY)

    def column_list(self, x_only=False):
        "Return column names of dataframe as a list"
        cols = self.df.columns.tolist()
        if x_only:
            cols = [n for n in cols if n.startswith('x')]
            if len(cols) == 0:
                raise IndexError("Could not find any features (x)")
        return cols

    @property
    def size(self):
        return self.df.size

    @property
    def shape(self):
        return self.df.shape

    def __getitem__(self, index):
        "Data indexing"
        typidx = type(index)
        if isinstance(index, str):
            if index.startswith('era'):
                if len(index) < 4:
                    raise IndexError('length of era string index too short')
                return self.era_isin([index])
            else:
                if index in ('train', 'validation', 'test', 'live'):
                    return self.region_isin([index])
                elif index == 'tournament':
                    return self.region_isin(TOURNAMENT_REGIONS)
                else:
                    raise IndexError('string index not recognized')
        elif typidx is pd.Series or typidx is np.ndarray:
            idx = index
            return Data(self.df[idx])
        else:
            raise IndexError('indexing type not recognized')

    @property
    def loc(self):
        "indexing by row ids"
        return Loc(self)

    def __len__(self):
        "Number of rows"
        return self.df.__len__()

    def __eq__(self, other_data):
        "Check if data objects are equal (True) or not (False); order matters"
        return self.df.equals(other_data.df)

    def __add__(self, other_data):
        "concatenate two data objects that have no overlap in ids"
        return concat_data([self, other_data])

    def __repr__(self):

        if self.__len__() == 0:
            return ''

        t = []
        fmt = '{:<10}{:<}'

        # region
        r = self.unique_region(as_str=True)
        stats = ', '.join(r)
        t.append(fmt.format('region', stats))

        # ids
        t.append(fmt.format('rows', len(self)))

        # era
        e = self.unique_era(as_str=True)
        stats = '{}, [{}, {}]'.format(e.size, e[0], e[-1])
        t.append(fmt.format('era', stats))

        # x
        x = self.x
        stats = '{}, min {:.4f}, mean {:.4f}, max {:.4f}'
        stats = stats.format(x.shape[1], x.min(), x.mean(), x.max())
        t.append(fmt.format('x', stats))

        # y
        y = self.df.y
        stats = 'mean {:.6f}, fraction missing {:.4f}'
        stats = stats.format(y.mean(), y.isnull().mean())
        t.append(fmt.format('y', stats))

        return '\n'.join(t)


def load_data(file_path):
    "Load data object from hdf archive; return Data"
    df = pd.read_hdf(file_path, key=HDF_DATA_KEY)
    return Data(df)


def load_zip(file_path, verbose=False):
    "Load numerai dataset from zip archive; return Data"

    # load zip
    zf = zipfile.ZipFile(file_path)
    train = pd.read_csv(zf.open(TRAIN_FILE), header=0, index_col=0)
    tourn = pd.read_csv(zf.open(TOURNAMENT_FILE), header=0, index_col=0)

    # turn into single dataframe and rename columns
    df = pd.concat([train, tourn], axis=0)
    rename_map = {'data_type': 'region', 'target': 'y'}
    for i in range(1, 51):
        rename_map['feature' + str(i)] = 'x' + str(i)
    df.rename(columns=rename_map, inplace=True)

    # convert era and region strings to np.float64
    df['era'] = df['era'].map(ERA_STR_TO_FLOAT)
    df['region'] = df['region'].map(REGION_STR_TO_FLOAT)

    # make sure memory is contiguous so that, e.g., data.x is a view
    df = df.copy()

    # to avoid copies we need the dtype of each column to be the same
    if df.dtypes.unique().size != 1:
        raise TypeError("dtype of each column should be the same")

    data = Data(df)
    if verbose:
        print(data)

    return data


def concat_data(datas):
    "Concatenate list-like of data objects; ids must not overlap"
    dfs = [d.df for d in datas]
    try:
        df = pd.concat(dfs, verify_integrity=True, copy=True)
    except ValueError:
        # pandas doesn't raise expected IndexError and for our large data
        # object, the id overlaps that it prints can be very long so
        raise IndexError("Overlap in ids found")
    return Data(df)


def compare_data(data1, data2, regions=None, n_jobs=1):
    """
    Compare two data objects, e.g., when they are from different datasets.

    The features, x, from the first dataset `data1` is used to fit a KNN tree.
    The nearest neighbor (k=1) of each row of features in `data2` is then
    found using the tree.

    `x distance` is the mean distance between the row of features in `data2`
    and its nearest neighbor row in `data1`.

    `y accuracy` is the fraction of rows for which the target y in `data2`
    matches the target of its nearest neighbor in `data1`.

    `era accuracy` is similar to `y accuracy` but for era instead of y.

    `d1-d2 rows` is the number of rows in `data1` minus the number of rows in
    `data2`.
    """
    if regions is None:
        regions = ('train', 'validation', 'test', 'live')
    df = pd.DataFrame(columns=regions)
    nn = NearestNeighbors(n_neighbors=1,  n_jobs=n_jobs)
    for region in regions:
        d1 = data1[data1.region == region]
        d2 = data2[data2.region == region]
        nn.fit(d1.x)
        dist, idx = nn.kneighbors(d2.x, n_neighbors=1, return_distance=True)
        idx = idx.reshape(-1)
        y1 = d1.y[idx]
        y2 = d2.y
        if np.isnan(y1).any() or np.isnan(y2).any():
            y_acc = np.nan
        else:
            y_acc = (y1 == y2).mean()
        x_dist = dist.mean()
        era_acc = (d1.era_float[idx] == d2.era_float).mean()
        df.loc['x distance', region] = x_dist
        df.loc['y accuracy', region] = y_acc
        df.loc['era accuracy', region] = era_acc
        df.loc['d1-d2 rows', region] = len(d1) - len(d2)
    return df


class Loc(object):
    "Utility class for the loc method."

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return Data(self.data.df.loc[index])
