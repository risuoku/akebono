from akebono.logging import getLogger
from .base import WrappedModel
from .sklearn import get_wrapped_sklearn_model
from akebono.io.operation.dumper import dump_sklearn_model
from akebono.io.operation.loader import load_sklearn_model
from akebono.utils import pathjoin
import akebono.settings as settings
import re
import copy

logger = getLogger(__name__)


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
        if model._pos_index is None:
            raise Exception('predict_proba need pos_index')
        return model._value.predict_proba(X)[:, model._pos_index]
     
    dump = dump_sklearn_model
    load = load_sklearn_model
    

def get_model(model_config):
    if not isinstance(model_config, dict):
        raise TypeError('model_config must be dict.')
    mcc = copy.copy(model_config)
    if 'name' not in mcc:
        raise Exception('name must be set in model_config.')
    model_name = mcc.pop('name')
    is_rebuild = mcc.pop('is_rebuild')
    scenario_tag = mcc.pop('scenario_tag', None)
    train_id = mcc.pop('train_id', None)

    model = None
    if re.search('^Sklearn.+$', model_name) is not None:
        model = get_wrapped_sklearn_model(model_name)(**mcc)
    elif model_name == 'LGBMClassifier':
        model = WrappedLGBMClassifier(**mcc)
    else:
        raise Exception('{} does not found.'.format(model_name))

    if model is None:
        raise Exception('unexpedted.')

    if is_rebuild:
        if scenario_tag is None or train_id is None:
            raise Exception('invalid state.')
        dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
        mname = 'train_result_model_{}'.format(train_id)
        model.load(dirpath, mname)
    logger.debug('get_model done in {} mode.'.format('predict' if is_rebuild else 'train'))
    return model
