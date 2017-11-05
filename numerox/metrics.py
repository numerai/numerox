import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


def metrics_per_era(data, pred_or_report, join='data'):

    df = pred_or_report.df

    # merge prediction or report with data (remove features x)
    if join == 'data':
        how = 'left'
    elif join == 'yhat':
        how = 'right'
    elif join == 'inner':
        how = 'inner'
    else:
        raise ValueError("`join` method not recognized")
    yhats_df = df.dropna()
    data_df = data.df[['era', 'region', 'y']]
    df = pd.merge(data_df, yhats_df, left_index=True, right_index=True,
                  how=how)

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
            m = calc_metrics_arrays(y, yhat)
            metrics[model].append(m)

    columns = ['logloss', 'auc', 'acc', 'ystd']
    for model in models:
        metrics[model] = pd.DataFrame(metrics[model], columns=columns,
                                      index=unique_eras)

    return metrics


def calc_metrics_arrays(y, yhat):
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
