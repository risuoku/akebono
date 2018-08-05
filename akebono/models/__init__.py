from akebono.logging import getLogger
from .base import WrappedModel
from akebono.io.operation.dumper import dump_sklearn_model
import os
import copy

logger = getLogger(__name__)


class WrappedLGBMClassifier(WrappedModel):
    def base_init_finished(self):
        if not self._is_rebuild:
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
        return self._value.predict_proba(X)
     
    dump = dump_sklearn_model
    

class WrappedRandomForestClassifier(WrappedModel):
    def base_init_finished(self):
        if not self._is_rebuild:
            self.reset()
    
    def fit(self, X, y):
        self._value.fit(X, y, **self._fit_kwargs)
        return self
    
    def reset(self):
        from sklearn.ensemble import RandomForestClassifier
        self._value = RandomForestClassifier(**self._init_kwargs)
    
    def predict(self, X):
        return self._value.predict(X)
    
    def predict_proba(self, X):
        return self._value.predict_proba(X)
     
    dump = dump_sklearn_model


class WrappedLogisticRegression(WrappedModel):
    def base_init_finished(self):
        if not self._is_rebuild:
            self.reset()
    
    def fit(self, X, y):
        self._value.fit(X, y, **self._fit_kwargs)
        return self
    
    def reset(self):
        from sklearn.linear_model import LogisticRegression
        self._value = LogisticRegression(**self._init_kwargs)
    
    def predict(self, X):
        return self._value.predict(X)
    
    def predict_proba(self, X):
        return self._value.predict_proba(X)
     
    dump = dump_sklearn_model
    

def get_model(model_config):
    if not isinstance(model_config, dict):
        raise TypeError('model_config must be dict.')
    mcc = copy.copy(model_config)
    if 'name' not in mcc:
        raise Exception('name must be set in model_config.')
    model_name = mcc.pop('name')
    mcc['is_rebuild'] = False

    if model_name == 'lgbm_classifier':
        return WrappedLGBMClassifier(**mcc)
    elif model_name == 'logistic_regression':
        return WrappedLogisticRegression(**mcc)
    elif model_name == 'random_forest_classifier':
        return WrappedRandomForestClassifier(**mcc)
    else:
        raise Exception('{} does not found.'.format(model_name))
