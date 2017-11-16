
=============
Release Notes
=============

These are the major changes made in each release. For details of the changes
see the commit log at http://github.com/kwgoodman/numerox

numerox 0.0.2
=============

*Release date: 2017-11-16*

The Data class is now a much lighter wrapper around the Numerai dataset.
Accessing the data with ``data.x`` and ``data.y`` now gives a view instead of
an expensive copy. To accomplish this the data is now stored in a contiguous
block of memory. To get contiguous memory all data now have the same data type
which was accomplished by converting era and region to floats.

Accessing the features ``data.x`` has dropped from 0.1 seconds to
0.0001 seconds.

``data.era`` and ``data.region`` still return numpy string arrays. For the
adventureous, you can get a view of the underlying floats by using
``data.era_float`` and ``data.region_float``.

HDF5 datasets (``data.save``) created with numerox 0.0.1 cannot be loaded with
0.0.2. To create new datasets, load from zip (``nx.load_zip``) and save.

numerox 0.0.1
=============

*Release date: 2017-11-14*

- Preview release of numerox
