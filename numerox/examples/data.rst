Data class
==========

The Data class is one of three main objects in numerox.

Download data
-------------

Let's download the current Numerai dataset::

    >>> import numerox as nx
    >>> nx.download('numerai_dataset.zip')

Or::

    >>> data = nx.download_deluxe('numerai_dataset.zip')

The second example will retry the download if it fails and it loads and returns
the data object.

Load data
---------

You can create a data object from the zip archive provided by Numerai::

    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> data
    region    train, validation, test, live
    rows      636965
    era       178, [era1, eraX]
    x         50, min 0.0000, mean 0.5025, max 1.0000
    y         mean 0.499546, fraction missing 0.3093

But that is slow (~7 seconds) which is painful for dedicated overfitters.
Let's convert the zip archive to an HDF5 archive::

    >>> data.save('numerai_dataset.hdf')
    >>> data2 = nx.load_data('numerai_dataset.hdf')

That loads quickly (~0.1 seconds, but takes more disk space than the
unexpanded zip archive).

Where's the data?
-----------------

To get views (not copies) of the data as numpy arrays use ``data.x`` and
``data.y[:]``. To get copies (not views) of ids, era, and region as numpy
string arrays use ``data.ids``, ``data.era``, ``data.region``.

Internally era and region are stored as floats. To get views of era and region
as numpy float arrays use ``data.era_float``, ``data.region_float``.

You can also request the targets, y, as a Pandas DataFrame::

    >>> data.y_df
                      bernie  elizabeth  jordan  ken  charles
    n2b2e3dd163cb422     1.0        1.0     1.0  1.0      1.0
    n177021a571c94c8     0.0        0.0     0.0  0.0      0.0
    <snip>

Here's how to get the targets of the 'elizabeth' tournament as NumPy arrays::

    data.y['elizabeth']
    data.y[2]

Indexing
--------

I'm going to show you a lot of indexing examples. If you are new to numerox
don't worry. You do not need to know them to get started.

Data indexing is done by rows, not columns::

    >>> data[data.y['bernie'] == 0]
    region    train, validation
    rows      220021
    era       132, [era1, era132]
    x         50, min 0.0000, mean 0.5025, max 1.0000
    y         mean 0.050645, fraction missing 0.0000

You can also index with special strings. Here are two examples::

    >>> data['era92']
    region    train
    rows      3370
    era       1, [era92, era92]
    x         50, min 0.0383, mean 0.5025, max 0.9885
    y         mean 0.499585, fraction missing 0.0000

    >>> data['tournament']
    region    validation, test, live
    rows      243352
    era       58, [era121, eraX]
    x         50, min 0.0000, mean 0.5026, max 1.0000
    y         mean 0.499638, fraction missing 0.8095

If you wish to extract more than one era::

    >>> data.era_isin(['era92', 'era93'])
    region    train
    rows      6956
    era       2, [era92, era93]
    x         50, min 0.0243, mean 0.5026, max 0.9885
    y         mean 0.499655, fraction missing 0.0000

You can do the same with regions::

    >>> data.region_isin(['test', 'live'])
    region    test, live
    rows      196990
    era       46, [era133, eraX]
    x         50, min 0.0000, mean 0.5026, max 1.0000
    y         mean nan, fraction missing 1.0000

Or you can remove regions (or eras)::

    >>> data.region_isnotin(['test', 'live'])
    region    train, validation
    rows      439975
    era       132, [era1, era132]
    x         50, min 0.0000, mean 0.5025, max 1.0000
    y         mean 0.499546, fraction missing 0.0000

You can concatenate data objects (as long as the ids don't overlap) by
adding them together. Let's add validation era121 to the training data::

    >>> data['train'] + data['era121']
    region    train, validation
    rows      397397
    era       121, [era1, era121]
    x         50, min 0.0000, mean 0.5025, max 1.0000
    y         mean 0.499535, fraction missing 0.0000

Or, let's go crazy::

    >>> nx.concat_data([data['live'], data['era1'], data['era92']])
    region    live, train
    rows      9403
    era       3, [eraX, era92]
    x         50, min 0.0000, mean 0.5025, max 0.9951
    y         mean 0.499482, fraction missing 0.4663

You can also index by Numerai row ids::

    >>> ids = ['n2b2e3dd163cb422', 'n177021a571c94c8', 'n7830fa4c0cd8466']
    >>> data.loc[ids]
    region    train
    rows      3
    era       1, [era1, era1]
    x         50, min 0.1675, mean 0.5077, max 0.8898
    y         mean 0.333333, fraction missing 0.0000

Slicing
-------

You can slice a Data object by era::

    >>> data['era90':'era120']
    region    train
    rows      111587
    era       31, [era90, era120]
    x         50, min 0.0000, mean 0.5026, max 1.0000
    y         mean 0.499578, fraction missing 0.0000

Note that the slice is inclusive of the ending era.

Why so many y's?
----------------

Correlation between the tournament targets::

    >>> data.y_df.corr()
                 bernie  elizabeth    jordan       ken   charles
    bernie     1.000000   0.806894  0.829468  0.933892  0.919436
    elizabeth  0.806894   1.000000  0.734084  0.795488  0.789388
    jordan     0.829468   0.734084  1.000000  0.816844  0.814362
    ken        0.933892   0.795488  0.816844  1.000000  0.895667
    charles    0.919436   0.789388  0.814362  0.895667  1.000000

Fraction of times pairwise targets are equal::

    >>> data.y_similarity()
                 bernie  elizabeth    jordan       ken   charles
    bernie     1.000000   0.903447  0.914734  0.966946  0.959718
    elizabeth  0.903447   1.000000  0.867042  0.897744  0.894694
    jordan     0.914734   0.867042  1.000000  0.908422  0.907181
    ken        0.966946   0.897744  0.908422  1.000000  0.947833
    charles    0.959718   0.894694  0.907181  0.947833  1.000000

Historgram of sum of targets across tournaments::

    >>> data.y_sum_hist()
          fraction
    ysum
    0     0.409678
    1     0.063760
    2     0.027231
    3     0.027390
    4     0.062367
    5     0.409573

Feature engineering
-------------------

Numerox offers several ways to transform features (``data.x``).

You can use principal component analysis (PCA) to make the features
orthogonal::

    >>> data2 = data.pca()

You can keep only the number of orthogonal features that explain at least,
say, 90% of the variance::

    >>> data2 = data.pca(nfactor=0.9)

which for the dataset I am using leaves me with 16 features (I'd get the
same result if I had used ``nfactor=16``):::

    >>> data2.xshape
    (636835, 16)

You can fit the PCA on, say, the train data and then use that fit to transform
all the data::

    >>> data2 = data.pca(nfactor=0.9, data_fit=data['train'])

Besides using PCA you can make your own (secret) transformations of the
features. Let's multiply all features by 2::

    >>> x = 2 * data.x
    >>> data2 = data.xnew(x)

Let's only keep the first 20 features::

    >>> x = data.x[:, :20]
    >>> data2 = data.xnew(x)

Let's double the number of features::

    >>> x = data.x
    >>> x = np.hstack((x, x * x))
    >>> data2 = data.xnew(x)

OK, you get the idea.


Try it
------

Numerox comes with a small dataset to play with::

    >>> nx.play_data()
    region    train, validation, test, live
    rows      6290
    era       178, [era1, eraX]
    x         50, min 0.0196, mean 0.5025, max 1.0000
    y         mean 0.504170, fraction missing 0.3099

It is about 1% of a regular Numerai dataset. The targets (``data.y``) are not
balanced.  It was created using the following function::

    play_data = data.subsample(fraction=0.01, seed=0)

If you have a long-running model then you can use subsample to create a
small dataset to quickly check that your code runs without crashing before
leaving it to run overnight.
