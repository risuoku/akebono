#from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from akebono.logging import getLogger
from sklearn.ensemble import RandomForestClassifier
from .base import WrappedModel
from .exporter import export_sklearn_model
import os

logger = getLogger(__name__)


    

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
     
    export = export_sklearn_model
    

class WrappedRandomForestClassifier(WrappedModel):
    def base_init_finished(self):
        if not self._is_rebuild:
            self.reset()
    
    def fit(self, X, y, fit_kwargs):
        self._value.fit(X, y, **fit_kwargs)
        return self
    
    def reset(self):
        self._value = RandomForestClassifier(**self._init_kwargs)
    
    def predict(self, X):
        return self._value.predict(X)
    
    def predict_proba(self, X):
        return self._value.predict_proba(X)
     
    export = export_sklearn_model

            
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
     
    export = export_sklearn_model
    

def get(kind, is_rebuild=False, init_kwargs={}):
    if kind == 'lgbm_classifier':
        return WrappedLGBMClassifier(is_rebuild, init_kwargs)
    elif kind == 'logistic_regression':
        return WrappedLogisticRegression(is_rebuild, init_kwargs)
    elif kind == 'random_forest_classifier':
        return WrappedRandomForestClassifier(is_rebuild, init_kwargs)
    else:
        raise Exception('{} does not found.'.format(kind))
