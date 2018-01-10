Data class
==========

The Data class is the most important object in numerox.

Load data quickly
-----------------

You can create a data object from the zip archive provided by Numerai::

    >>> import numerox as nx
    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> data
    region    live, test, train, validation
    rows      884544
    era       98, [era1, eraX]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.3109

But that is slow (~6 seconds) which is painful for dedicated overfitters.
Let's create an HDF5 archive::

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

Indexing
--------

I'm going to show you a lot of indexing examples. If you are new to numerox
don't worry. You do not need to know them to get started.

Data indexing is done by rows, not columns::

    >>> data[data.y == 0]
    region    train, validation
    rows      304813
    era       97, [era1, era97]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.000000, fraction missing 0.0000

You can also index with special strings. Here are two examples::

    >>> data['era92']
    region    validation
    rows      6048
    era       1, [era92, era92]
    x         50, min 0.0308, mean 0.4993, max 1.0000
    y         mean 0.500000, fraction missing 0.0000

    >>> data['tournament']
    region    live, test, validation
    rows      348831
    era       13, [era86, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean 0.499966, fraction missing 0.7882

If you wish to extract more than one era (I hate these eras)::

    >>> data.era_isin(['era92', 'era93'])
    region    validation
    rows      12086
    era       2, [era92, era93]
    x         50, min 0.0177, mean 0.4993, max 1.0000
    y         mean 0.500000, fraction missing 0.0000

You can do the same with regions::

    >>> data.region_isin(['test', 'live'])
    region    live, test
    rows      274966
    era       1, [eraX, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean nan, fraction missing 1.0000

Or you can remove regions (or eras)::

    >>> data.region_isnotin(['test', 'live'])
    region    train, validation
    rows      609578
    era       97, [era1, era97]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.0000

You can concatenate data objects (as long as the ids don't overlap) by
adding them together. Let's add validation era92 to the training data::

    >>> data['train'] + data['era92']
    region    train, validation
    rows      541761
    era       86, [era1, era92]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499960, fraction missing 0.0000

Or, let's go crazy::

    >>> nx.concat_data([data['live'], data['era1'], data['era92']])
    region    live, train, validation
    rows      19194
    era       3, [era1, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean 0.499960, fraction missing 0.3544

You can also index by Numerai row ids::

    >>> data = nx.play_data()
    >>> ids = data.ids[10:12]
    >>> ids
    array(['ne04fa6947fd4485', 'nfa60f9e305a34a8'],
          dtype='|S16')
    >>> data.loc[ids]
    region    train
    rows      2
    era       1, [era1, era1]
    x         50, min 0.2088, mean 0.4964, max 0.7633
    y         mean 0.500000, fraction missing 0.0000

Try it
------

Numerox comes with a small dataset to play with::

    >>> nx.play_data()
    region    train, validation, test, live
    rows      5690
    era       133, [era1, eraX]
    x         50, min 0.0106, mean 0.5025, max 0.9855
    y         mean 0.500000, fraction missing 0.3466

It is about 1% of a regular Numerai dataset. The data (``data.y``) is balanced.
It was created using the following function::

    play_data = data.subsample(fraction=0.01, balance=True, seed=0)

If you have a long-running model then you can use subsample to create a
small dataset to quickly check that your code runs without crashing before
leaving it to run overnight.
