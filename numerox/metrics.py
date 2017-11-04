import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


def calc_metrics(data, df):

    # merge prediction with data (remove features x)
    yhats_df = df.dropna()
    data_df = data.df[['era', 'region', 'y']]
    df = pd.merge(data_df, yhats_df, left_index=True, right_index=True,
                  how='left')

    models = yhats_df.columns.values
    metrics = {}
    for model in models:
        metrics[model] = []

    # calc metrics for each era
    unique_eras = df.era.unique()
    for era in unique_eras:
        idx = df.era.isin([era])
        df_era = df[idx]
        y = df_era['y'].values
        for model in models:
            yhat = df_era[model].values
            m = _calc_metrics_1era(y, yhat)
            metrics[model].append(m)

    columns = ['logloss', 'auc', 'acc', 'ystd']
    for model in models:
        metrics[model] = pd.DataFrame(metrics[model], columns=columns,
                                      index=unique_eras)

    return metrics


def _calc_metrics_1era(y, yhat):
    "standard metrics for `yhat` array given actual outcome `y` array"

    metrics = []

    # logloss
    try:
        m = log_loss(y, yhat)
    except ValueError:
        m = np.nan
    metrics.append(m)

    # auc
    try:
        m = roc_auc_score(y, yhat)
    except ValueError:
        m = np.nan
    metrics.append(m)

    # acc
    yh = np.zeros(yhat.size)
    yh[yhat >= 0.5] = 1
    try:
        m = accuracy_score(y, yh)
    except ValueError:
        m = np.nan
    metrics.append(m)

    # std(yhat)
    metrics.append(yhat.std())

    return metrics
