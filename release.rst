
=============
Release Notes
=============

- v0.6.0

  * Add ``ten99``
  * Add ``top_stakers``
  * Add ``download_earnings``
  * Add ``nmr_at_addr``
  * Add ``token_price_data``
  * Add ``historical_price``

- v0.5.0

  * Add ``compare_data``
  * Add ``show_stakes`` example
  * ``prediction.save`` now has an append mode
  * ``prediction.performance`` now returns a dataframe
  * ``prediction.dominance`` now returns a dataframe
  * ``prediction.summary`` now returns a dataframe
  * Remove ``prediction.performance_df``
  * Remove ``prediction.dominance_df``
  * Remove ``prediction.summary_df``
  * ``prediction.performance`` keyword changed from cols to columns
  * More prediction methods can now handle empty predictions

- v0.4.0

  * Numerox requires NumerAPI 0.8.1
  * Add ``merge_predictions``
  * Add ``prediction.check``
  * Add more columns to ``prediction.originality``
  * Add optional choice of columns in ``prediction.performance_df``
  * Column name changed to 'concord' in ``concordance`` function
  * Add example comparing performance of a single change across models
  * Add ``data.y_to_nan``
  * Bugfix: prediction.__repr__ sometimes showed wrong fraction of missing y
  * Bugfix: merging predictions with same name but different ids may fail
  * More unit tests

- v0.3.1

  * Unit test coverage 89%, up from 65%
  * Improve reporting of ``upload`` status
  * Add ``testing.HiddenPrints``
  * Bug fix: checking for equality of empty Predictions crashes
  * ``get_stakes`` informative error message when round_number < 61
  * More unit tests

- v0.3.0

  * Add ``upload`` to make submissions
  * ``download`` is the new name for ``download_dataset``
  * Add ``prediction.compare``
  * Add ``prediction.loc`` for indexing by Numerai row ids
  * Add ``prediction.rename``
  * Add ``prediction.drop``
  * ``prediction.concordance`` 3x faster when prediction contains 10 names
  * ``prediction.concordance`` now sorts by concordance
  * ``prediction.merge`` is no longer an inplace operation
  * ``prediction.merge_arrays`` is no longer an inplace operation
  * No longer take ``name`` as input: ``prediction.to_csv``,
    ``prediction.summary``, ``prediction.summary_df``
  * Remove ``model.hash``
  * Examples can now be run after installation: nx.examples.run_all_examples()
  * Redo compare_models example
  * Make more use of numerapi
  * Python package requests is no longer a dependency
  * Rewrite ``testing.micro_prediction`` for better unit testing
  * More unit tests

- v0.2.0

  * This release makes a large change to the numerox API
  * There are now 3 main classes instead of 4
  * The Report class has been merged into the Prediction class
  * The Prediction class can now hold the predictions from multiple models
  * New features have been added to the Prediction class

- v0.1.2

  * Numerox now uses (and requires) NumerAPI
  * Add file overwrite protection option to ``download_dataset``
  * Beware: ``download_dataset`` will now raise by default if file exists
  * Add ``report.__setitem__``
  * Add ``report.__contains__``
  * Add ``data.loc`` for indexing by Numerai row ids
  * Add ``report.originality``
  * Add report indexing (``report.__getitem__``)
  * More unit tests

- v0.1.1

  * Complete rewrite of all performance metrics
  * Add ``metrics_per_model``
  * Add ``report.dominance``
  * Add ``report.dominance_df``
  * Add ``prediction.performance_df``
  * Add ``mlpc`` model
  * Remove ``xgboost`` model to remove optional xgboost dependency
  * Rewrite examples of comparing performance of multiple models
  * More unit tests

- v0.1.0

  * Add ``report.correlation``
  * Add ``prediction.consistency``
  * Add ``prediction.metrics_per_era``
  * Can now specify which metrics to calculate in ``metrics_per_era``
  * Add sort_by to ``show_stakes``
  * Add ``prediction.yhatnew``
  * Add ``xgboost`` model
  * Add ``randomforest`` model
  * Add ``logisticPCA`` model
  * Models at top level: ``nx.extatrees`` instead of nx.model.extratrees, etc
  * ``logistic`` model now uses less regularization by default
  * Bugfix: display model name correctly when parameter dictionary is empty
  * More unit tests

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
