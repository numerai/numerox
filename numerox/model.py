import json
from typing import List

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression

try:
    from xgboost.sklearn import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

"""

Make your own model
-------------------

First take a look at the logistic and extratrees models below.

Your model MUST have a fit_predict method that takes two data objects as
input. The first is training data, the second is prediction data.

The fit_predict method MUST return two numpy arrays. The first contains the
ids, the second the predictions. Make sure that these two arrays stay aligned!

The models below inherit from The Model class. That is optional. But if you do
inherit from Model and if you place your parameters in self.p as is done in
the models below then you will get a nice printout (model name and parameters)
when you run your model.

OK, now go make money!

"""


class Model(object):

    def hash(self, data_fit, data_predict):
        """"
        Hash of data, model name, and parameters dictionary if you have one.

        And if you are hashing it is a good idea to have a parameters
        dictionary.
        """
        h = []
        h.append(data_fit.hash())
        h.append(data_predict.hash())
        h.append(self.__class__.__name__)
        if hasattr(self, "p"):
            h.append(json.dumps(self.p, sort_keys=True))
        h = tuple(h)
        return hash(h)

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
        self.clf = None

    def fit_predict(self, data_fit, data_predict):
        self.fit(data_fit.x, data_fit.y)
        yhat = self.predict_proba(data_predict.x)[:, 1]
        return data_predict.ids, yhat

    def get_model(self):
        return LogisticRegression(C=self.p['inverse_l2'])

    def fit(self, x, y):
        if self.clf is None:
            self.clf = self.get_model()
        self.clf.fit(x, y)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)


class pipeline(Model):
    def __init__(self, models):
        self.models = [(model.__class__.__name__, model) for model in models]

    def fit_predict(self, data_fit, data_predict):
        from sklearn.pipeline import Pipeline
        model = Pipeline(steps=self.models)
        model.fit(data_fit.x, data_fit.y)
        yhat = model.predict_proba(data_predict.x)[:, 1]
        return data_predict.ids, yhat


class extratrees(Model):

    def __init__(self, ntrees=100, depth=3, nfeatures=7, seed=0):
        self.p = {'ntrees': ntrees,
                  'depth': depth,
                  'nfeatures': nfeatures,
                  'seed': seed}

    def fit_predict(self, data_fit, data_predict):
        clf = ETC(criterion='gini',
                  max_features=self.p['nfeatures'],
                  max_depth=self.p['depth'],
                  n_estimators=self.p['ntrees'],
                  random_state=self.p['seed'],
                  n_jobs=-1)
        clf.fit(data_fit.x, data_fit.y)
        yhat = clf.predict_proba(data_predict.x)[:, 1]
        return data_predict.ids, yhat


class randomforest(Model):

    def __init__(self, ntrees=100, depth=3, max_features=2, seed=0):
        self.p = {'ntrees': ntrees,
                  'depth': depth,
                  'max_features': max_features,
                  'seed': seed}

    def fit_predict(self, data_fit, data_predict):
        clf = RFC(criterion='gini',
                  max_features=self.p['max_features'],
                  max_depth=self.p['depth'],
                  n_estimators=self.p['ntrees'],
                  random_state=self.p['seed'],
                  n_jobs=-1)
        clf.fit(data_fit.x, data_fit.y)
        yhat = clf.predict_proba(data_predict.x)[:, 1]
        return data_predict.ids, yhat


class xgboost(Model):

    def __init__(self, learning_rate=0.1, subsample=0.4, max_depth=5,
                 n_estimators=5, seed=0):
        self.p = {'learning_rate': learning_rate,
                  'subsample': subsample,
                  'max_depth': max_depth,
                  'n_estimators': n_estimators,
                  'seed': seed}
        if not HAS_XGBOOST:
            raise ImportError("You must install xgboost to use this model")

    def fit_predict(self, data_fit, data_predict):
        clf = XGBClassifier(learning_rate=self.p['learning_rate'],
                            subsample=self.p['subsample'],
                            max_depth=self.p['max_depth'],
                            n_estimators=self.p['n_estimators'],
                            seed=self.p['seed'],
                            nthread=-1)
        clf.fit(data_fit.x, data_fit.y)
        yhat = clf.predict_proba(data_predict.x)[:, 1]
        return data_predict.ids, yhat


# fast model for testing; always predicts 0.5
class fifty(Model):

    def __init__(self):
        self.p = {}

    def fit_predict(self, data_fit, data_predict):
        yhat = 0.5 * np.ones(len(data_predict))
        return data_predict.ids, yhat
