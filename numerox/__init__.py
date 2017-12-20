# flake8: noqa

# classes
from numerox.data import Data
from numerox.prediction import Prediction
from numerox.report import Report

# models
from numerox.model import Model
from numerox.model import logistic
from numerox.model import extratrees
from numerox.model import randomforest
from numerox.model import mlpc
from numerox.model import logisticPCA

# load
from numerox.data import load_data, load_zip
from numerox.testing import play_data
from numerox.prediction import load_prediction
from numerox.report import load_report

# splitters
from numerox.splitter import TournamentSplitter
from numerox.splitter import ValidationSplitter
from numerox.splitter import SplitSplitter
from numerox.splitter import CheatSplitter
from numerox.splitter import CVSplitter
from numerox.splitter import IgnoreEraCVSplitter
from numerox.splitter import RollSplitter

# run
from numerox.run import production
from numerox.run import backtest
from numerox.run import run
from numerox.run import Runner

# numerai
from numerox.numerai import download_dataset
from numerox.numerai import dataset_url

# misc
from numerox.data import concat_data
from numerox.metrics import concordance
from numerox.metrics import metrics_per_era
from numerox.metrics import metrics_per_model
from numerox.numerai import show_stakes
from numerox.version import __version__

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No numerox unit testing available")
