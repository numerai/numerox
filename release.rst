
=============
Release Notes
=============

- v0.0.4 (not yet released)

  * Add ``data.pca``
  * Add examples of transforming features
  * You can now change the number of features with ``data.x_replace``
  * ``data.replace_x`` renamed to ``data.x_replace``
  * ``shares_memory`` can now check datas with different number of x columns
  * Add more unit tests

- v0.0.3

  * Add examples
  * Add iterator ``data.era_iter``
  * Add iterator ``data.region_iter``
  * ``prediction.ids`` and ``prediction.yhat`` are now views instead of copies
  * Remove appveyor so that unit tests can use Python's tempfile
  * Bugfix: ``prediction.copy`` was not copying the index
  * Bugfix: mistakes in two unit tests meant they could never fail
  * Add more unit tests

- v0.0.2

  * ``data.x`` and ``data.y`` now return fast views instead of slow copies
  * era and region stored internally as floats
  * HDF5 datasets created with v0.0.1 cannot be loaded with v0.0.2

- v0.0.1

  * Preview release of numerox
