Data class
==========

The Data class is the most important object in numerox.

Load data quickly
-----------------

You can create a data object from the zip archive provided by Numerai::

    >>> import numerox as nx
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
``data.y``. To get copies (not views) of ids, era, and region as numpy
string arrays use ``data.ids``, ``data.era``, ``data.region``.

Internally era and region are stored as floats. To get views of era and region
as numpy float arrays use ``data.era_float``, ``data.region_float``.

Here are some ways to look the targets of the 'elizabeth' tournament::

    data.y2
    data.y_for_tournament('elizabeth')
    data.y_for_tournament(2)

Indexing
--------

I'm going to show you a lot of indexing examples. If you are new to numerox
don't worry. You do not need to know them to get started.

Data indexing is done by rows, not columns::

    >>> data[data.y1 == 0]
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
adding them together. Let's add validation era92 to the training data::

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
    y         mean 0.333333, fraction missing 0.000

Why so many y's?
----------------

Correlation between the tournament targets::

    >>> data.y_correlation()
              y1        y5        y2        y3        y4
    y1  1.000000  0.919436  0.806894  0.829468  0.933892
    y5  0.919436  1.000000  0.789388  0.814362  0.895667
    y2  0.806894  0.789388  1.000000  0.734084  0.795488
    y3  0.829468  0.814362  0.734084  1.000000  0.816844
    y4  0.933892  0.895667  0.795488  0.816844  1.000000

Fraction of times pairwise targets are equal::

    >>> data.y_similarity()
              y1        y2        y3        y4        y5
    y1  1.000000  0.959718  0.903447  0.914734  0.966946
    y2  0.959718  1.000000  0.894694  0.907181  0.947833
    y3  0.903447  0.894694  1.000000  0.867042  0.897744
    y4  0.914734  0.907181  0.867042  1.000000  0.908422
    y5  0.966946  0.947833  0.897744  0.908422  1.000000

Historgram of sum of targets across tournaments::

    >>> data.y_sum_hist()
          fraction
    ysum
    0     0.409678
    1     0.063760
    2     0.027231
    3     0.027390

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
