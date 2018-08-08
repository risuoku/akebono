import re

from .base import WrappedModel
from akebono.io.operation.dumper import dump_sklearn_model
from akebono.io.operation.loader import load_sklearn_model
from akebono.utils import load_object_by_str
from akebono.logging import getLogger


logger = getLogger(__name__)


_valid_models = {
    'LogisticRegression': 'sklearn.linear_model',
    'RandomForestClassifier': 'sklearn.ensemble',
}


def _fit(model, X, y):
    model._value.fit(X, y, **model._fit_kwargs)
    return model


def _predict(model, X):
    return model._value.predict(X)


def _predict_proba(model, X):
    if model._pos_index is None:
        raise Exception('predict_proba need pos_index')
    return model._value.predict_proba(X)[:, model._pos_index]


def _generate_reset_func(cls):

    def _func(model):
        model._value = cls(**model._init_kwargs)
    return _func


def _base_init_finished(model):
    model.reset()


def get_wrapped_sklearn_model(model_cls_str):
    # model_cls_str は'^Sklearn.+$' にマッチする前提
    model_cls_str = re.sub('Sklearn', '', model_cls_str)
    model_cls_mod = _valid_models.get(model_cls_str)
    if model_cls_mod is None:
        raise Exception('{} is invalid.'.format(model_cls_str))
    model_cls = load_object_by_str('{}@{}'.format(model_cls_str, model_cls_mod))

    cls_attrs = {
        'fit': _fit,
        'reset': _generate_reset_func(model_cls),
        'dump': dump_sklearn_model,
        'load': load_sklearn_model,
        'base_init_finished': _base_init_finished,
    }

    if hasattr(model_cls, 'predict'):
        cls_attrs['predict'] = _predict
    if hasattr(model_cls, 'predict_proba'):
        cls_attrs['predict_proba'] = _predict_proba
    logger.debug('model_cls_str: {} .. created.'.format(model_cls_str))

    return type('WrappedSklearn{}'.format(model_cls_str), (WrappedModel,), cls_attrs)
