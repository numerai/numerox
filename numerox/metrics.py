import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

import numerox as nx
from numerox.data import ERA_INT_TO_STR
from numerox.data import REGION_INT_TO_STR

LOGLOSS_BENCHMARK = 0.693


def metrics_per_era(data, prediction, tournament, join='data',
                    columns=['logloss', 'auc', 'acc', 'ystd'],
                    era_as_str=False, region_as_str=False, split_pairs=True):
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
    cols = ['era', 'region'] + nx.tournament_all(as_str=True)
    data_df = data.df[cols]
    df = pd.merge(data_df, yhats_df, left_index=True, right_index=True,
                  how=how)

    regions = df['region'].unique().tolist()
    if region_as_str:
        regions = [REGION_INT_TO_STR[r] for r in regions]

    # calc metrics for each era
    pairs = yhats_df.columns.values
    metrics = []
    unique_eras = df.era.unique()
    for era in unique_eras:
        idx = df.era.isin([era])
        df_era = df[idx]
        if era_as_str:
            era = ERA_INT_TO_STR[era]
        for pair in pairs:
            if tournament is None:
                # evaluate with targets that model trained on
                tourni = nx.tournament_str(pair[1])
            else:
                # force evaluation targets to be from given tournament
                tourni = nx.tournament_str(tournament)
            y = df_era[tourni].values
            yhat = df_era[pair].values
            m = calc_metrics_arrays(y, yhat, columns)
            m = [era, pair] + m
            metrics.append(m)

    columns = ['era', 'pair'] + columns
    metrics = pd.DataFrame(metrics, columns=columns)

    if split_pairs:
        metrics = add_split_pairs(metrics)

    return metrics, regions


def metrics_per_name(data, prediction, tournament, join='data',
                     columns=['logloss', 'auc', 'acc', 'ystd'],
                     era_as_str=True, region_as_str=True, split_pairs=True):

    # calc metrics per era
    skip = ['sharpe', 'consis']
    cols = [c for c in columns if c not in skip]
    if 'sharpe' in columns or 'consis' in columns:
        if 'logloss' not in cols:
            cols.append('logloss')
    mpe, regions = metrics_per_era(data, prediction, tournament, join=join,
                                   columns=cols)

    # gather some info
    info = {}
    info['era'] = mpe['era'].unique().tolist()
    info['region'] = regions
    if era_as_str:
        info['era'] = [ERA_INT_TO_STR[e] for e in info['era']]
    if region_as_str:
        info['region'] = [REGION_INT_TO_STR[r] for r in info['region']]

    if 'logloss' in cols:
        # pivot is a dataframe with:
        #     era for rows
        #     pair for columns
        #     logloss for cell values
        pivot = mpe.pivot(index='era', columns='pair', values='logloss')

    # mm is a dataframe with:
    #    pair as rows
    #    `cols` as columns
    mm = mpe.groupby('pair').mean()

    # metrics is the output with:
    #    pair as rows
    #    `columns` as columns
    metrics = pd.DataFrame(index=mm.index, columns=columns)

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

    if split_pairs:
        metrics = add_split_pairs(metrics)

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
        elif col == 'ymin':
            m = yhat.min()
        elif col == 'ymax':
            m = yhat.max()
        elif col == 'ymean':
            m = yhat.mean()
        elif col == 'ystd':
            m = yhat.std()
        elif col == 'length':
            m = yhat.size
        else:
            raise ValueError("unknown metric ({})".format(col))
        metrics.append(m)
    return metrics


def concordance(data, prediction, split_pairs=True):
    "Concordance; less than 0.12 is passing; data should be the full dataset."

    pairs = prediction.pairs(as_str=False)
    concords = pd.DataFrame(columns=['concord'], index=pairs)

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
    for i in range(len(pairs)):
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
        concords.iloc[i] = concord

    concords = concords.sort_values('concord')

    if split_pairs:
        concords = add_split_pairs(concords)

    return concords


def add_split_pairs(df, as_str=True):
    "Add name and tournament columns and optional drop pair column"
    if 'pair' in df:
        pairs = df['pair'].tolist()
    else:
        pairs = df.index.tolist()
    name, tournament = zip(*pairs)
    if as_str:
        tournament = [nx.tournament_str(t) for t in tournament]
    df.insert(0, 'tournament', tournament)
    df.insert(0, 'name', name)
    return df
