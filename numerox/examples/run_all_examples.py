import inspect
import numerox as nx


def run_all_examples(data=None):
    "Run most of the numerox examples"

    if data is None:
        data = nx.numerai.download_data_object(verbose=True)

    backtest_example = nx.examples.backtest_example
    print_source(backtest_example)
    backtest_example(data)

    concordance_example = nx.examples.concordance_example
    print_source(concordance_example)
    concordance_example(data)

    compare_models = nx.examples.compare_models
    print_source(compare_models)
    compare_models(data)

    compare_change = nx.examples.compare_change
    print_source(compare_change)
    compare_change(data)

    cv_warning = nx.examples.cv_warning
    print_source(cv_warning)
    cv_warning(data['train'], nsamples=2)


def print_source(func):
    print('-' * 70)
    print('\n{}\n'.format(func.__name__.upper()))
    lines = inspect.getsourcelines(func)
    print("".join(lines[0]))
