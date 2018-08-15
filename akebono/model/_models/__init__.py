from .base import WrappedModel
from .sklearn import get_wrapped_sklearn_model
from akebono.io.operation.dumper import dump_sklearn_model
from akebono.io.operation.loader import load_sklearn_model


class WrappedLGBMClassifier(WrappedModel):
    def base_init_finished(self):
        self.reset()
    
    def fit(self, X, y):
        self._value.fit(X, y, **self._fit_kwargs)
        return self
    
    def reset(self):
        from lightgbm import LGBMClassifier
        self._value = LGBMClassifier(**self._init_kwargs)
    
    def predict(self, X):
        return self._value.predict(X)
    
    def predict_proba(self, X):
        if self._pos_index is None:
            raise Exception('predict_proba need pos_index')
        return self._value.predict_proba(X)[:, self._pos_index]
     
    dump = dump_sklearn_model
    load = load_sklearn_model
