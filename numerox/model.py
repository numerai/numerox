import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

"""

Make your own model
-------------------

First take a look at the logistic regression model below (named logistic). The
model is just a thin wrapper around sklearn's LogisticRegression. The wrapper
allows LogisticRegression to receive data from numerox and for numerox to keep
track of its predictions.

Your model MUST have a fit_predict method that takes two data objects as
input. The first is training data (dfit), the second is prediction data (dpre).

The fit_predict method MUST return two numpy arrays. The first contains the
ids, the second the predictions. Make sure that these two arrays stay aligned!

The models below inherit from The Model class. That is optional. But if you do
inherit from Model and if you place your parameters in self.p as is done in
the models below then you will get a nice printout (model name and parameters)
when you run your model.

None of the models below will be competitive in the Numerai tournament. You'll
have to make your own model. If you already have a model then you can make a
thin wrapper around it, as is done below, to get it to run with numerox.

OK, now go make money!

"""


class Model(object):

    def __repr__(self):
        msg = ""
        model = self.__class__.__name__
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


class logistic(Model):

    def __init__(self, inverse_l2=0.0001):
        self.p = {'inverse_l2': inverse_l2}

    def fit_predict(self, dfit, dpre):
        model = LogisticRegression(C=self.p['inverse_l2'])
        model.fit(dfit.x, dfit.y)
        yhat = model.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat


class extratrees(Model):

    def __init__(self, ntrees=100, depth=3, nfeatures=7, seed=0):
        self.p = {'ntrees': ntrees,
                  'depth': depth,
                  'nfeatures': nfeatures,
                  'seed': seed}

    def fit_predict(self, dfit, dpre):
        clf = ETC(criterion='gini',
                  max_features=self.p['nfeatures'],
                  max_depth=self.p['depth'],
                  n_estimators=self.p['ntrees'],
                  random_state=self.p['seed'],
                  n_jobs=-1)
        clf.fit(dfit.x, dfit.y)
        yhat = clf.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat


class randomforest(Model):

    def __init__(self, ntrees=100, depth=3, max_features=2, seed=0):
        self.p = {'ntrees': ntrees,
                  'depth': depth,
                  'max_features': max_features,
                  'seed': seed}

    def fit_predict(self, dfit, dpre):
        clf = RFC(criterion='gini',
                  max_features=self.p['max_features'],
                  max_depth=self.p['depth'],
                  n_estimators=self.p['ntrees'],
                  random_state=self.p['seed'],
                  n_jobs=-1)
        clf.fit(dfit.x, dfit.y)
        yhat = clf.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat


class mlpc(Model):

    def __init__(self, alpha=0.11, layers=[5, 3], activation='tanh',
                 learn=0.002, seed=0):
        self.p = {'alpha': alpha,
                  'layers': layers,
                  'activation': activation,
                  'learn': learn,
                  'seed': seed}

    def fit_predict(self, dfit, dpre):
        clf = MLPC(hidden_layer_sizes=self.p['layers'],
                   alpha=self.p['alpha'],
                   activation=self.p['activation'],
                   learning_rate_init=self.p['learn'],
                   random_state=self.p['seed'],
                   max_iter=200)
        clf.fit(dfit.x, dfit.y)
        yhat = clf.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat


# model used by numerai to generate example_predictions.csv
class example_predictions(Model):

    def __init__(self):
        self.p = {}

    def fit_predict(self, dfit, dpre):
        model = GradientBoostingClassifier(n_estimators=25, max_depth=1,
                                           random_state=1776)
        model.fit(dfit.x, dfit.y)
        yhat = model.predict_proba(dpre.x)[:, 1]
        yhat = np.round(yhat, 5)
        return dpre.ids, yhat


# sklearn pipeline example
class logisticPCA(Model):

    def __init__(self, nfeatures=10, inverse_l2=1e-4):
        self.p = {'inverse_l2': inverse_l2,
                  'nfeatures': nfeatures}

    def fit_predict(self, dfit, dpre):
        pipe = Pipeline([('pca', PCA(n_components=self.p['nfeatures'])),
                         ("lr", LogisticRegression(C=self.p['inverse_l2']))])
        pipe.fit(dfit.x, dfit.y)
        yhat = pipe.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat


# fast model for testing; always predicts 0.5
class fifty(Model):

    def __init__(self):
        self.p = {}

    def fit_predict(self, dfit, dpre):
        yhat = 0.5 * np.ones(len(dpre))
        return dpre.ids, yhat
