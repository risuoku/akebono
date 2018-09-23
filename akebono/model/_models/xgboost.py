from .base import WrappedModel
from akebono.utils import pathjoin


class WrappedXGBClassifier(WrappedModel):
    def base_init_finished(self):
        self.reset()
    
    def fit(self, X, y):
        self._value.fit(X, y, **self._fit_kwargs)
        return self
    
    def reset(self):
        from xgboost import XGBClassifier
        self._value = XGBClassifier(**self._init_kwargs)
    
    def predict(self, X):
        return self._value.predict(X)
    
    def predict_proba(self, X):
        if self._pos_index is None:
            raise Exception('predict_proba need pos_index')
        return self._value.predict_proba(X)[:, self._pos_index]

    def dump(self, dirpath, name):
        self.value.save_model(pathjoin(dirpath, name + '.bin'))
        return self

    def load(self, dirpath, name):
        self._value.load_model(pathjoin(dirpath, name + '.bin'))
        return self
