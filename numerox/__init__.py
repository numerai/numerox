# flake8: noqa

# classes
from numerox.data import Data
from numerox.prediction import Prediction
from numerox.report import Report
from numerox.model import Model

# models
import numerox.model

# load
from numerox.data import load_data, load_zip
from numerox.testing import play_data
from numerox.prediction import load_prediction
from numerox.report import load_report

# splitters
from numerox.splitter import tournament_splitter
from numerox.splitter import validation_splitter
from numerox.splitter import split_splitter
from numerox.splitter import cv_splitter

# run
from numerox.run import run
from numerox.run import production
from numerox.run import backtest

# numerai
from numerox.numerai import download_dataset

# misc
from numerox.data import concat_data
from numerox.metrics import metrics_per_era
from numerox.version import __version__

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No numerox unit testing available")
