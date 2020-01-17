import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor as MLPC
from sklearn.ensemble import ExtraTreesRegressor as ETC
from sklearn.ensemble import RandomForestRegressor as RFC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import numerox as nx
"""

Here are the directions for making your own model:

https://github.com/kwgoodman/numerox/blob/master/numerox/examples/model.rst

"""

# ---------------------------------------------------------------------------
# base class for all models


class Model(object):
    def __repr__(self):
        model = self.name
        msg = ""
        if hasattr(self, "p"):
            if len(self.p) == 0:
                msg += model + "()"
            else:
                msg += model + "("
                for name, value in self.p.items():
                    msg += name + "=" + str(value) + ", "
                msg = msg[:-2]
                msg += ")"
        else:
            msg += model + "()"
        return msg

    @property
    def name(self):
        "Name of model"
        if not hasattr(self, '_name'):
            self._name = self.__class__.__name__
        return self._name

    def rename(self, name):
        "Rename model in place; model is returned"
        if name is None:
            return self
        if not nx.isstring(name):
            raise ValueError('`name` must be a string')
        self._name = name
        return self


# ---------------------------------------------------------------------------
# numerox example models


class linear(Model):
    def fit_predict(self, dfit, dpre, tournament):
        model = LinearRegression()
        model.fit(dfit.x, dfit.y[tournament])
        yhat = model.predict(dpre.x)
        return dpre.ids, yhat


class ridge_mean(Model):
    def __init__(self, alpha=6):
        self.p = {'alpha': alpha}

    def fit_predict(self, dfit, dpre, tournament):
        model = Ridge(alpha=self.p['alpha'], normalize=True)
        yfit = dfit.y[:].mean(axis=1)
        model.fit(dfit.x, yfit)
        yhat = model.predict(dpre.x)
        return dpre.ids, yhat


class extratrees(Model):
    def __init__(self, ntrees=100, depth=3, nfeatures=7, seed=0):
        self.p = {
            'ntrees': ntrees,
            'depth': depth,
            'nfeatures': nfeatures,
            'seed': seed
        }

    def fit_predict(self, dfit, dpre, tournament):
        clf = ETC(criterion='mse',
                  max_features=self.p['nfeatures'],
                  max_depth=self.p['depth'],
                  n_estimators=self.p['ntrees'],
                  random_state=self.p['seed'],
                  n_jobs=-1)
        clf.fit(dfit.x, dfit.y[tournament])
        yhat = clf.predict(dpre.x)
        return dpre.ids, yhat


class randomforest(Model):
    def __init__(self, ntrees=100, depth=3, max_features=2, seed=0):
        self.p = {
            'ntrees': ntrees,
            'depth': depth,
            'max_features': max_features,
            'seed': seed
        }

    def fit_predict(self, dfit, dpre, tournament):
        clf = RFC(criterion='mse',
                  max_features=self.p['max_features'],
                  max_depth=self.p['depth'],
                  n_estimators=self.p['ntrees'],
                  random_state=self.p['seed'],
                  n_jobs=-1)
        clf.fit(dfit.x, dfit.y[tournament])
        yhat = clf.predict(dpre.x)
        return dpre.ids, yhat


class mlpc(Model):
    def __init__(self,
                 alpha=0.11,
                 layers=[5, 3],
                 activation='tanh',
                 learn=0.002,
                 seed=0):
        self.p = {
            'alpha': alpha,
            'layers': layers,
            'activation': activation,
            'learn': learn,
            'seed': seed
        }

    def fit_predict(self, dfit, dpre, tournament):
        clf = MLPC(hidden_layer_sizes=self.p['layers'],
                   alpha=self.p['alpha'],
                   activation=self.p['activation'],
                   learning_rate_init=self.p['learn'],
                   random_state=self.p['seed'],
                   max_iter=200)
        clf.fit(dfit.x, dfit.y[tournament])
        yhat = clf.predict(dpre.x)
        return dpre.ids, yhat


# model used by numerai to generate example_predictions.csv
class example_predictions(Model):
    def __init__(self):
        self.p = {}

    def fit_predict(self, dfit, dpre, tournament):
        model = GradientBoostingRegressor(n_estimators=25,
                                          max_depth=1,
                                          random_state=1776)
        model.fit(dfit.x, dfit.y[tournament])
        yhat = model.predict(dpre.x)
        yhat = np.round(yhat, 5)
        return dpre.ids, yhat


# sklearn pipeline example
class linearPCA(Model):
    def __init__(self, nfeatures=10):
        self.p = {'nfeatures': nfeatures}

    def fit_predict(self, dfit, dpre, tournament):
        pipe = Pipeline([('pca', PCA(n_components=self.p['nfeatures'])),
                         ("lr", LinearRegression())])
        pipe.fit(dfit.x, dfit.y[tournament])
        yhat = pipe.predict(dpre.x)
        return dpre.ids, yhat


# fast model for testing; always predicts 0.5
class fifty(Model):
    def __init__(self):
        self.p = {}

    def fit_predict(self, dfit, dpre, tournament):
        yhat = 0.5 * np.ones(len(dpre))
        return dpre.ids, yhat
