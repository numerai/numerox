
=============
Release Notes
=============

- v0.0.9

  * Add ability to work with new (round 85) Numerai datasets
  * Update ``play_data`` with new numerai dataset
  * ``run`` now hides from your model the y you are trying to predict
  * Cumsum in ``show_stakes`` and ``get_stakes`` now dollars above you
  * ``model.hash`` combined hash of data, model name, and model parameters
  * Gentle refactor of splitters to reuse code
  * Bugfix: crash when balancing already balanced data
  * More unit tests

- v0.0.8

  * Add ``show_stakes``
  * Add ``get_stakes``
  * ``data.xnew`` is 3 times faster
  * ``data.column_list(x_only=False)`` replaces _column_list and _x_names
  * Example of Numerai's cross validation warning (hold out eras not rows)
  * Bugfix: ``data.xnew`` output didn't use contiguous memory

- v0.0.7

  * Add ``data.balance``
  * Add ``data.subsample``
  * Add ``data.hash``
  * Add ``IgnoreEraCVSplitter``
  * Add ``dataset_url`` function
  * All splitters now use a single base class
  * Add ``download_data_object`` to avoid hard coding path in examples
  * ``play_data`` is now ``data.y`` balanced
  * Rewrote ``update_play_data``
  * More unit tests

- v0.0.6

  * Add ``concordance``
  * New Runner class can run multiple models through a single data splitter
  * Update ``download_dataset`` for recent Numerai API change
  * Add ``RollSplitter`` roll forward fit-predict splits from consecutive eras
  * Add another verbosity level to ``run`` (verbosity=3)
  * Use ``play_data`` instead of numerai server or hard coding my local path
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
