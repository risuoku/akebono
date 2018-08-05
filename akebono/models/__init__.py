from akebono.logging import getLogger
from .base import WrappedModel
from .sklearn import get_wrapped_sklearn_model
from akebono.io.operation.dumper import dump_sklearn_model
import os
import re
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
    

def get_model(model_config):
    if not isinstance(model_config, dict):
        raise TypeError('model_config must be dict.')
    mcc = copy.copy(model_config)
    if 'name' not in mcc:
        raise Exception('name must be set in model_config.')
    model_name = mcc.pop('name')
    mcc['is_rebuild'] = False

    if re.search('^Sklearn.+$', model_name) is not None:
        return get_wrapped_sklearn_model(model_name)(**mcc)
    elif model_name == 'LGBMClassifier':
        return WrappedLGBMClassifier(**mcc)
    else:
        raise Exception('{} does not found.'.format(model_name))
