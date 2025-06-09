
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.callbacks import EarlyStopping

class MyKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, batch_size=32, epochs=10, verbose=0):
        self.build_fn = build_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model_ = None
        self._build_params = {}

    def set_params(self, **params):
        keras_keys = ["batch_size", "epochs", "verbose", "build_fn"]
        for key in list(params.keys()):
            if key not in keras_keys:
                self._build_params[key] = params.pop(key)
        for k in keras_keys:
            setattr(self, k, params.get(k, getattr(self, k, None)))
        return self

    def get_params(self, deep=True):
        return {
            "build_fn": self.build_fn,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "verbose": self.verbose,
            **self._build_params
        }

    def fit(self, X, y):
        features = X.shape[-1]
        self.model_ = self.build_fn(features=features, **self._build_params)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                        verbose=self.verbose, callbacks=[EarlyStopping(patience=2)])
        return self

    def predict(self, X):
        return (self.model_.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        return self.model_.predict(X)
