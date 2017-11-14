import zipfile

import numpy as np
import pandas as pd

TRAIN_FILE = 'numerai_training_data.csv'
TOURNAMENT_FILE = 'numerai_tournament_data.csv'
HDF_DATA_KEY = 'numerox_data'
TOURNAMENT_REGIONS = ['validation', 'test', 'live']


class Data(object):

    def __init__(self, df):
        self.df = df

    # ids -------------------------------------------------------------------

    @property
    def ids(self):
        "Copy of ids as a numpy str array"
        return self.df.index.values.astype(str)

    # era -------------------------------------------------------------------

    @property
    def era(self):
        "Copy of era as a 1d numpy str array"
        return self.df.era.values.astype(str)

    def unique_era(self):
        "array of unique eras"
        return self.df.era.unique().astype(str)

    def era_isin(self, eras):
        "Copy of data containing only eras in the iterable `eras`"
        idx = self.df.era.isin(eras)
        return self[idx]

    def era_isnotin(self, eras):
        "Copy of data containing eras that are not the iterable `eras`"
        idx = self.df.era.isin(eras)
        return self[~idx]

    # region ----------------------------------------------------------------

    @property
    def region(self):
        "Copy of region as a 1d numpy str array"
        return self.df.region.values.astype(str)

    def unique_region(self):
        "array of unique regions"
        return self.df.region.unique().astype(str)

    def region_isin(self, regions):
        "Copy of data containing only regions in the iterable `regions`"
        idx = self.df.region.isin(regions)
        return self[idx]

    def region_isnotin(self, regions):
        "Copy of data containing regions that are not the iterable `regions`"
        idx = self.df.region.isin(regions)
        return self[~idx]

    # x ---------------------------------------------------------------------

    @property
    def x(self):
        "Copy of features, x, as a numpy array"
        names = self._x_names()
        return self.df[names].values

    def replace_x(self, x_array):
        "Copy of data but with data.x=`x_array`; must have same number of rows"
        df = self.df.copy()
        xname = self._x_names()
        df[xname] = x_array
        return Data(df)

    def _x_names(self):
        "Return list of column names of features, x, in dataframe"
        cols = self._column_list()
        names = [n for n in cols if n.startswith('x')]
        if len(names) == 0:
            raise IndexError("Could not find any features (x)")
        return names

    @property
    def xshape(self):
        "Shape (nrows, ncols) of x"
        rows = self.df.shape[0]
        cols = len(self._x_names())
        return (rows, cols)

    # y ---------------------------------------------------------------------

    @property
    def y(self):
        "Copy of y as a 1d numpy array"
        return self.df['y'].values.copy()

    def copy(self):
        "Copy of data"
        return Data(self.df.copy(deep=True))

    def save(self, path_or_buf, compress=False):
        "Save data as an hdf archive"
        if compress:
            self.df.to_hdf(path_or_buf, HDF_DATA_KEY,
                           complib='zlib', complevel=4)
        else:
            self.df.to_hdf(path_or_buf, HDF_DATA_KEY)

    def _column_list(self):
        "Return column names of dataframe as a list"
        return self.df.columns.tolist()

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
        r = self.unique_region()
        stats = ', '.join(r)
        t.append(fmt.format('region', stats))

        # ids
        t.append(fmt.format('rows', len(self)))

        # era
        e = self.unique_era()
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


def load_zip(file_path):
    "Load numerai dataset from zip archive; return Data"
    zf = zipfile.ZipFile(file_path)
    train = pd.read_csv(zf.open(TRAIN_FILE), header=0, index_col=0)
    tourn = pd.read_csv(zf.open(TOURNAMENT_FILE), header=0, index_col=0)
    df = pd.concat([train, tourn], axis=0)
    rename_map = {'data_type': 'region', 'target': 'y'}
    for i in range(1, 51):
        rename_map['feature' + str(i)] = 'x' + str(i)
    df.rename(columns=rename_map, inplace=True)
    return Data(df)


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
