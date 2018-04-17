import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from numerox.data import ERA_INT_TO_STR
from numerox.data import REGION_INT_TO_STR

LOGLOSS_BENCHMARK = 0.693


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
            m = (pivot < LOGLOSS_BENCHMARK).mean(axis=0)
        elif col == 'sharpe':
            m = (LOGLOSS_BENCHMARK - pivot).mean(axis=0) / pivot.std(axis=0)
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
        elif col == 'logloss_pass':
            try:
                m = log_loss(y, yhat) < LOGLOSS_BENCHMARK
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
        elif col == 'length':
            m = yhat.size
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
            d = [ks_2samp(yhat0, yhat1)[0],
                 ks_2samp(yhat0, yhat2)[0],
                 ks_2samp(yhat2, yhat1)[0]]
            ks.append(max(d))
        concord = np.mean(ks)
        concords.loc[name] = concord

    concords = concords.sort_values('concord')

    return concords
