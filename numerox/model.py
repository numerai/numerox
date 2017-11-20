import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier as ETC

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

    def __repr__(self):
        msg = ""
        model = self.__class__.__name__
        msg += model + "("
        if hasattr(self, "p"):
            for name, value in self.p.items():
                msg += name + "=" + str(value) + ", "
            msg = msg[:-2]
            msg += ")"
        else:
            msg += model + "()"
        return msg


class logistic(Model):

    def __init__(self, inverse_l2=0.00001):
        self.p = {'inverse_l2': inverse_l2}

    def fit_predict(self, data_fit, data_predict):
        model = LogisticRegression(C=self.p['inverse_l2'])
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
                  random_state=self.p['seed'])
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
