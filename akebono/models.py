#from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from akebono.logging import getLogger
from akebono.utils import to_pickle
import akebono.settings as settings
import os

logger = getLogger(__name__)


def _export_sklearn_model(obj, model_name):
    return to_pickle(
        os.path.join(settings.models_dir, '{}.pkl'.format(model_name)),
        obj.value)
    

class WrappedModel:
    def __init__(self, is_rebuild, init_kwargs):
        self._init_kwargs = init_kwargs
        self._is_rebuild = is_rebuild
        self._value = None
        self.base_init_finished()
    
    def base_init_finished(self):
        pass
    
    def fit(self, X, y, fit_kwargs):
        raise NotImplementedError()
    
    def reset(self):
        raise NotImplementedError()
    
    def export(self, name):
        raise NotImplementedError()
    
    def rebuild(self):
        if not self._is_rebuild:
            raise Exception('rebuild flag is False')
    
    @property
    def value(self):
        return self._value

            
class WrappedLGBMClassifier(WrappedModel):
    def base_init_finished(self):
        if not self._is_rebuild:
            self.reset()
    
    def fit(self, X, y, fit_kwargs):
        self._value.fit(X, y, **fit_kwargs)
        return self
    
    def reset(self):
        self._value = LGBMClassifier(**self._init_kwargs)
    
    def predict(self, X):
        return self._value.predict(X)
    
    def predict_proba(self, X):
        return self._value.predict_proba(X)
     
    export = _export_sklearn_model

            
class WrappedLogisticRegression(WrappedModel):
    def base_init_finished(self):
        if not self._is_rebuild:
            self.reset()
    
    def fit(self, X, y, fit_kwargs):
        self._value.fit(X, y, **fit_kwargs)
        return self
    
    def reset(self):
        self._value = LogisticRegression(**self._init_kwargs)
    
    def predict(self, X):
        return self._value.predict(X)
    
    def predict_proba(self, X):
        return self._value.predict_proba(X)
     
    export = _export_sklearn_model
    

def get(kind, is_rebuild=False, init_kwargs={}):
    if kind == 'lgbm_classifier':
        return WrappedLGBMClassifier(is_rebuild, init_kwargs)
    elif kind == 'logistic_regression':
        return WrappedLogisticRegression(is_rebuild, init_kwargs)
    else:
        raise Exception('{} does not found.'.format(kind))
