import inspect
import numerox as nx


def run_all_examples(data=None):
    "Run most of the numerox examples"

    if data is None:
        data = nx.numerai.download_data_object(verbose=True)

    backtest = nx.examples.backtest
    print_source(backtest)
    backtest(data)

    concordance = nx.examples.concordance
    print_source(concordance)
    concordance(data)

    improve_model = nx.examples.improve_model
    print_source(improve_model)
    improve_model(data)

    cv_warning = nx.examples.cv_warning
    print_source(cv_warning)
    cv_warning(nx.logistic(), data['train'], nsamples=2)


def print_source(func):
    print('-' * 70)
    print('\n{}\n'.format(func.__name__.upper()))
    lines = inspect.getsourcelines(func)
    print("".join(lines[0]))
