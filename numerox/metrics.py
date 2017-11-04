import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


def calc_metrics(data, df):

    # merge prediction with data (remove features x)
    yhats_df = df.dropna()
    data = data.region_isin(['train', 'validation'])  # speed optimzation
    data_df = data.df[['era', 'region', 'y']]  # speed optimzation
    df = pd.merge(data_df, yhats_df, left_index=True, right_index=True,
                  how='right')

    models = yhats_df.columns.values
    metrics = {}
    for model in models:
        metrics[model] = {'train': [], 'validation': []}

    for region in ('train', 'validation'):

        # pull out region
        idx = df.region.isin([region])
        df_region = df[idx]
        if len(df_region) == 0:
            continue

        # calc metrics for each era
        unique_eras = df_region.era.unique()
        for era in unique_eras:
            idx = df_region.era.isin([era])
            df_era = df_region[idx]
            y = df_era['y'].values
            for model in models:
                yhat = df_era[model].values
                m = _calc_metrics_1era(y, yhat)
                metrics[model][region].append(m)

        columns = ['logloss', 'auc', 'acc', 'ystd']
        for model in models:
            metrics[model][region] = pd.DataFrame(metrics[model][region],
                                                  columns=columns,
                                                  index=unique_eras)

    return metrics


def _calc_metrics_1era(y, yhat):
    "standard metrics for `yhat` array given actual outcome `y` array"
    m = []
    m.append(log_loss(y, yhat))
    m.append(roc_auc_score(y, yhat))
    yh = np.zeros(yhat.size)
    yh[yhat >= 0.5] = 1
    m.append(accuracy_score(y, yh))
    m.append(yhat.std())
    return m
