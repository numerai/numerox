
=============
Release Notes
=============

- v0.0.6 (not yet released)

  * Update ``download_dataset`` for recent Numerai API change
  * Bugfix: in v0.0.5 CVSplitter ran only a single cross validation fold
  * More unit tests

- v0.0.5

  * Data splitters can now be reused to run more than one model
  * To reuse a splitter, reset it: ``splitter.reset()``
  * All splitters renamed; e.g. ``cheat_splitter`` is now ``CheatSplitter``
  * Splitters are now iterator classes instead of generator functions
  * ``data.ids`` returns numpy string array copy instead of object array view
  * More unit tests

- v0.0.4

  * Add ``data.pca``
  * Add examples of transforming features
  * You can now change the number of features with ``data.xnew``
  * ``data.xnew`` is the new name of ``data.replace_x``
  * ``shares_memory`` can now check datas with different number of x columns
  * More unit tests

- v0.0.3

  * Add examples
  * Add iterator ``data.era_iter``
  * Add iterator ``data.region_iter``
  * ``prediction.ids`` and ``prediction.yhat`` are now views instead of copies
  * Remove appveyor so that unit tests can use Python's tempfile
  * Bugfix: ``prediction.copy`` was not copying the index
  * Bugfix: mistakes in two unit tests meant they could never fail
  * More unit tests

- v0.0.2

  * ``data.x`` and ``data.y`` now return fast views instead of slow copies
  * era and region stored internally as floats
  * HDF5 datasets created with v0.0.1 cannot be loaded with v0.0.2

- v0.0.1

  * Preview release of numerox
