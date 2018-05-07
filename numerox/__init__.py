# flake8: noqa

# classes
from numerox.data import Data
from numerox.prediction import Prediction

# models
from numerox.model import Model
from numerox.model import logistic
from numerox.model import extratrees
from numerox.model import randomforest
from numerox.model import mlpc
from numerox.model import example_predictions
from numerox.model import logisticPCA

# load
from numerox.data import load_data
from numerox.data import load_zip
from numerox.testing import play_data
from numerox.prediction import load_prediction
from numerox.prediction import load_prediction_csv
from numerox.prediction import load_example_predictions

# splitters
from numerox.splitter import TournamentSplitter
from numerox.splitter import ValidationSplitter
from numerox.splitter import SplitSplitter
from numerox.splitter import CheatSplitter
from numerox.splitter import CVSplitter
from numerox.splitter import LoocvSplitter
from numerox.splitter import IgnoreEraCVSplitter
from numerox.splitter import RollSplitter

# run
from numerox.run import production
from numerox.run import backtest
from numerox.run import run

# numerai
from numerox.numerai import download
from numerox.numerai import upload
from numerox.numerai import round_resolution_date
from numerox.numerai import year_to_round_range

# report
from numerox.report import Report

# leaderboard
from numerox.leaderboard import Leaderboard
from numerox.leaderboard import get_current_round_number

# tokens
from numerox.tokens import nmr_at_addr
from numerox.tokens import token_price_data
from numerox.tokens import historical_price
from numerox.tokens import nmr_resolution_price

# misc
from numerox import examples
from numerox.data import concat_data
from numerox.data import compare_data
from numerox.numerai import show_stakes
from numerox.numerai import get_stakes
from numerox.numerai import is_controlling_capital
from numerox.version import __version__
from numerox.prediction import merge_predictions

# util
from numerox.util import isint
from numerox.util import isstring
from numerox.util import history

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No numerox unit testing available")
