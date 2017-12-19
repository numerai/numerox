import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

import numerox as nx
from numerox.data import ERA_INT_TO_STR
from numerox.data import REGION_INT_TO_STR


def metrics_per_era(data, pred_or_report, join='data',
                    columns=['logloss', 'auc', 'acc', 'ystd'],
                    era_as_str=False, region_as_str=False):
    "Dataframe with columns era, model, and specified metrics. And region list"

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

    regions = df['region'].unique().tolist()
    if region_as_str:
        regions = [REGION_INT_TO_STR[r] for r in regions]

    # calc metrics for each era
    models = yhats_df.columns.values
    metrics = []
    unique_eras = df.era.unique()
    for era in unique_eras:
        idx = df.era.isin([era])
        df_era = df[idx]
        y = df_era['y'].values
        if era_as_str:
            era = ERA_INT_TO_STR[era]
        for model in models:
            yhat = df_era[model].values
            m = calc_metrics_arrays(y, yhat, columns)
            m = [era, model] + m
            metrics.append(m)

    columns = ['era', 'model'] + columns
    metrics = pd.DataFrame(metrics, columns=columns)

    return metrics, regions


def metrics_per_model(data, report, join='data',
                      columns=['logloss', 'auc', 'acc', 'ystd'],
                      era_as_str=True, region_as_str=True):

    if not isinstance(report, nx.Report):
        raise TypeError("`report` must be a nx.Report object")

    # calc metrics per era
    skip = ['sharpe', 'consis']
    cols = [c for c in columns if c not in skip]
    if 'sharpe' in columns or 'consis' in columns:
        if 'logloss' not in cols:
            cols.append('logloss')
    mpe, regions = metrics_per_era(data, report, join=join, columns=cols)

    # gather some info
    info = {}
    info['era'] = mpe['era'].unique().tolist()
    info['region'] = regions
    if era_as_str:
        info['era'] = [ERA_INT_TO_STR[e] for e in info['era']]
    if region_as_str:
        info['region'] = [REGION_INT_TO_STR[r] for r in info['region']]

    # pivot is a dataframe with:
    #     era for rows
    #     model for columns
    #     logloss for cell values
    pivot = mpe.pivot(index='era', columns='model', values='logloss')

    # mm is a dataframe with:
    #    model as rows
    #    `cols` as columns
    mm = mpe.groupby('model').mean()

    # metrics is the output with:
    #    model as rows
    #    `columns` as columns
    metrics = pd.DataFrame(index=pivot.columns, columns=columns)

    for col in columns:
        if col == 'consis':
            m = (pivot < np.log(2)).mean(axis=0)
        elif col == 'sharpe':
            m = (np.log(2) - pivot).mean(axis=0) / pivot.std(axis=0)
        elif col == 'logloss':
            m = mm['logloss']
        elif col == 'auc':
            m = mm['auc']
        elif col == 'acc':
            m = mm['acc']
        elif col == 'ystd':
            m = mm['ystd']
        else:
            raise ValueError("unknown metric ({})".format(col))
        metrics[col] = m

    return metrics, info


def calc_metrics_arrays(y, yhat, columns):
    "standard metrics for `yhat` array given actual outcome `y` array"
    metrics = []
    for col in columns:
        if col == 'logloss':
            try:
                m = log_loss(y, yhat)
            except ValueError:
                m = np.nan
        elif col == 'auc':
            try:
                m = roc_auc_score(y, yhat)
            except ValueError:
                m = np.nan
        elif col == 'acc':
            yh = np.zeros(yhat.size)
            yh[yhat >= 0.5] = 1
            try:
                m = accuracy_score(y, yh)
            except ValueError:
                m = np.nan
        elif col == 'ystd':
            m = yhat.std()
        else:
            raise ValueError("unknown metric ({})".format(col))
        metrics.append(m)
    return metrics


def concordance(data, prediction):
    "Concordance; less than 0.12 is passing; data should be the full dataset."

    # all data
    x = [data.x]
    yhat = [None]

    # data for each region
    regions = ['validation', 'test', 'live']
    for region in regions:
        d = data[region]
        x.append(d.x)
        yh = prediction.df.yhat[d.df.index].values  # align
        yhat.append(yh)

    # make clusters
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=1337)
    kmeans.fit(x[0])
    c1 = kmeans.predict(x[1])
    c2 = kmeans.predict(x[2])
    c3 = kmeans.predict(x[3])

    # cross cluster distance (KS distance)
    ks = []
    for i in set(c1):
        yhat1 = yhat[1][c1 == i]
        yhat2 = yhat[2][c2 == i]
        yhat3 = yhat[3][c3 == i]
        d = [ks_2samp(yhat1, yhat2),
             ks_2samp(yhat1, yhat3),
             ks_2samp(yhat3, yhat2)]
        ks.append(max(d))
    concord = np.mean(ks)

    return concord


# copied from scipy to avoid scipy dependency; modified for use in numerox
def ks_2samp(y1, y2):
    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.

    Parameters
    ----------
    y1, y2 : sequence of 1-D ndarrays
        two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different

    Returns
    -------
    statistic : float
        KS statistic

    Notes
    -----
    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample K-S test, the distribution is
    assumed to be continuous.

    This is the two-sided test, one-sided tests are not implemented.
    The test uses the two-sided asymptotic Kolmogorov-Smirnov distribution.

    If the K-S statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same.
    """
    y1 = np.sort(y1)
    y2 = np.sort(y2)
    n1 = y1.shape[0]
    n2 = y2.shape[0]
    data_all = np.concatenate([y1, y2])
    cdf1 = np.searchsorted(y1, data_all, side='right') / (1.0*n1)
    cdf2 = np.searchsorted(y2, data_all, side='right') / (1.0*n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    return d
