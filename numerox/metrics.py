import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans
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
