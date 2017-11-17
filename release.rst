
=============
Release Notes
=============

- v0.0.3

  * Added iterator ``data.era_iter``
  * Added iterator ``data.region_iter``
  * ``prediction.ids`` and ``prediction.yhat`` are now views instead of copies
  * Bugfix: ``prediction.copy`` was not copying the index
  * Remove appveyor
  * Add more unit tests

- v0.0.2

  * ``data.x`` and ``data.y`` now return fast views instead of slow copies
  * era and region stored internally as floats
  * HDF5 datasets created with v0.0.1 cannot be loaded with v0.0.2

- v0.0.1

  * Preview release of numerox
