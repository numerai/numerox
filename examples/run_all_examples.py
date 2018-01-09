# This is just a way for the developers of numerox to see if all examples run

import inspect
import numerox as nx

from backtest_example import backtest_example
from concordance_example import concordance_example
from compare_models import compare_models
from cv_warning import cv_warning


def print_source(func):
    lines = inspect.getsourcelines(func)
    print("".join(lines[0]))


if __name__ == '__main__':

    data = nx.numerai.download_data_object(verbose=True)

    print('-' * 70)
    print('\nBACKTEST EXAMPLE\n')
    print_source(backtest_example)
    backtest_example(data)

    print('-' * 70)
    print('\nCONCORDANCE EXAMPLE\n')
    print_source(concordance_example)
    concordance_example(data)

    print('-' * 70)
    print('\nCOMPARE MODELS\n')
    print_source(compare_models)
    compare_models(data)

    print('-' * 70)
    print('\nCV WARNING\n')
    print_source(cv_warning)
    cv_warning(data['train'], nsamples=2)
