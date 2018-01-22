import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from numerox.data import ERA_INT_TO_STR
from numerox.data import REGION_INT_TO_STR


def metrics_per_era(data, prediction, join='data',
                    columns=['logloss', 'auc', 'acc', 'ystd'],
                    era_as_str=False, region_as_str=False):
    "Dataframe with columns era, model, and specified metrics. And region list"

    df = prediction.df

    # merge prediction with data (remove features x)
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
    names = yhats_df.columns.values
    metrics = []
    unique_eras = df.era.unique()
    for era in unique_eras:
        idx = df.era.isin([era])
        df_era = df[idx]
        y = df_era['y'].values
        if era_as_str:
            era = ERA_INT_TO_STR[era]
        for name in names:
            yhat = df_era[name].values
            m = calc_metrics_arrays(y, yhat, columns)
            m = [era, name] + m
            metrics.append(m)

    columns = ['era', 'name'] + columns
    metrics = pd.DataFrame(metrics, columns=columns)

    return metrics, regions


def metrics_per_name(data, prediction, join='data',
                     columns=['logloss', 'auc', 'acc', 'ystd'],
                     era_as_str=True, region_as_str=True):

    # calc metrics per era
    skip = ['sharpe', 'consis']
    cols = [c for c in columns if c not in skip]
    if 'sharpe' in columns or 'consis' in columns:
        if 'logloss' not in cols:
            cols.append('logloss')
    mpe, regions = metrics_per_era(data, prediction, join=join, columns=cols)

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
    #     name for columns
    #     logloss for cell values
    pivot = mpe.pivot(index='era', columns='name', values='logloss')

    # mm is a dataframe with:
    #    name as rows
    #    `cols` as columns
    mm = mpe.groupby('name').mean()

    # metrics is the output with:
    #    name as rows
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

    concords = pd.DataFrame(columns=['concord'], index=prediction.names)

    # fit clusters
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=1337)
    kmeans.fit(data.x)

    # yhats and clusters for each region
    yhats = []
    clusters = []
    for region in ['validation', 'test', 'live']:
        d = data[region]
        cluster = kmeans.predict(d.x)
        clusters.append(cluster)
        yh = prediction.df.loc[d.df.index].values  # align
        yhats.append(yh)

    # cross cluster distance (KS distance)
    for i, name in enumerate(prediction.names):
        ks = []
        for j in set(clusters[0]):
            yhat0 = yhats[0][:, i][clusters[0] == j]
            yhat1 = yhats[1][:, i][clusters[1] == j]
            yhat2 = yhats[2][:, i][clusters[2] == j]
            d = [ks_2samp(yhat0, yhat1),
                 ks_2samp(yhat0, yhat2),
                 ks_2samp(yhat2, yhat1)]
            ks.append(max(d))
        concord = np.mean(ks)
        concords.loc[name] = concord

    concords = concords.sort_values('concord')

    return concords


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


# copied from scipy to avoid scipy dependency; modified for use in numerox
def pearsonr(x, y):
    """
    Calculate a Pearson correlation coefficient.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed, and not necessarily zero-mean.
    Like other correlation coefficients, this one varies between -1 and +1
    with 0 implying no correlation. Correlations of -1 or +1 imply an exact
    linear relationship. Positive correlations imply that as x increases, so
    does y. Negative correlations imply that as x increases, y decreases.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------
    r : float
        Pearson's correlation coefficient

    Notes
    -----

    The correlation coefficient is calculated as follows:

    .. math::

        r_{pb} = \frac{\sum (x - m_x) (y - m_y)
                       }{\sqrt{\sum (x - m_x)^2 (y - m_y)^2}}

    where :math:`m_x` is the mean of the vector :math:`x` and :math:`m_y` is
    the mean of the vector :math:`y`.


    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = max(min(r, 1.0), -1.0)

    return r
