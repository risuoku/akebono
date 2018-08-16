import sklearn.metrics as skl_metrics
from akebono.utils import (
    load_object_by_str,
    get_label_by_object,
)
from akebono.logging import getLogger


logger = getLogger(__name__)


_binary_classifier_metrics = [
    {
        'name': 'precision',
        'func': skl_metrics.precision_score,
        'predict_type': 'predict',
    },
    {
        'name': 'recall',
        'func': skl_metrics.recall_score,
        'predict_type': 'predict',
    },
    {
        'name': 'accuracy',
        'func': skl_metrics.accuracy_score,
        'predict_type': 'predict',
    },
    {
        'name': 'f1_score',
        'func': skl_metrics.f1_score,
        'predict_type': 'predict',
    },
    {
        'name': 'log_loss',
        'func': skl_metrics.log_loss,
        'predict_type': 'predict',
    },
    {
        'name': 'roc_auc',
        'func': skl_metrics.roc_auc_score,
        'predict_type': 'predict_proba',
    },
]


_multiple_classifier_metrics = [
    {
        'name': 'accuracy',
        'func': skl_metrics.accuracy_score,
        'predict_type': 'predict',
    },
]


_regressor_metrics = [
    {
        'name': 'mean_absolute_error',
        'func': skl_metrics.mean_absolute_error,
        'predict_type': 'predict',
    },
    {
        'name': 'mean_squared_error',
        'func': skl_metrics.mean_squared_error,
        'predict_type': 'predict',
    },
    {
        'name': 'median_absolute_error',
        'func': skl_metrics.median_absolute_error,
        'predict_type': 'predict',
    },
    {
        'name': 'r2_score',
        'func': skl_metrics.r2_score,
        'predict_type': 'predict',
    },
    {
        'name': 'explained_variance',
        'func': skl_metrics.explained_variance_score,
        'predict_type': 'predict',
    },
]


def _validate_metrics(o):
    if not isinstance(o, dict):
        raise TypeError('invalid type.')
    if ('name' not in o) or ('func' not in o) or ('predict_type' not in o):
        raise Exception('invalid key.')
    if o['predict_type'] not in ('predict', 'predict_proba',):
        raise Exception('invalid predict_type')


def _fit_and_predict(X_train, X_test, y_train, model):
    model.reset()
    model.fit(X_train, y_train)
    y_pred = y_pred_proba = None

    if hasattr(model, 'predict'):
        try:
            y_pred = model.predict(X_test)
        except NotImplementedError:
            # ignore exception
            pass
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)
        except NotImplementedError:
            # ignore exception
            pass
    
    return model, y_pred, y_pred_proba


def _get_evaluated_result(y_pred, y_pred_proba, y_test, metrics):
    _validate_metrics(metrics)

    if metrics['predict_type'] == 'predict':
        if y_pred is not None:
            return {
                'name': metrics['name'],
                'value': metrics['func'](y_test, y_pred),
            }
        else:
            return {
                'name': metrics['name'],
                'value': None,
            }
    else:
        if y_pred_proba is not None:
            return {
                'name': metrics['name'],
                'value': metrics['func'](y_test, y_pred_proba),
            }
        else:
            return {
                'name': metrics['name'],
                'value': None,
            }
    raise Exception('unexpected process.')


def _sklearn_cross_val_iter2train_test_iter(X, y, cross_val_iter, cross_val_iter_kwargs):
    for train_index, test_index in cross_val_iter(**cross_val_iter_kwargs).split(X.index):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        yield X_train, X_test, y_train, y_test


def evaluate(model,
    X, y, preprocessor,
    train_test_split_func='train_test_split@sklearn.model_selection',
    train_test_split_func_kwargs={},
    cross_val_iterator=None,
    cross_val_iterator_kwargs={},
    metrics='all'
    ):
    if model.model_type is None:
        model.set_model_type(y=y)
    model_type = model.model_type

    # cross_val_iteratorはNone or iterable
    if cross_val_iterator is not None:
        if isinstance(cross_val_iterator, str):
            cross_val_iterator = load_object_by_str(cross_val_iterator)
    else:
        # train_test_split_func はNone or string or function
        if isinstance(train_test_split_func, str):
            train_test_split_func = load_object_by_str(train_test_split_func)
    
    result = {'metrics': []}
    train_test_iterator = None

    # not CV mode
    if cross_val_iterator is None:
        train_test_iterator = [train_test_split_func(X, y, **train_test_split_func_kwargs)]
        result['cv'] = False
    # CV mode
    else:
        train_test_iterator = _sklearn_cross_val_iter2train_test_iter(X, y, cross_val_iterator, cross_val_iterator_kwargs)
        result['cv'] = True

    for Xraw_train, Xraw_test, y_train, y_test in train_test_iterator:
        preprocessor.reset()
        X_train, X_test = preprocessor.process(Xraw_train, Xraw_test)
        one_result = []
        model, y_pred, y_pred_proba = _fit_and_predict(X_train, X_test, y_train, model)
        if metrics == 'all':
            _preload_metrics = None
            if model_type == 'binary_classifier':
                _preload_metrics = _binary_classifier_metrics
            elif model_type == 'regressor':
                _preload_metrics = _regressor_metrics
            elif model_type == 'multiple_classifier':
                _preload_metrics =_multiple_classifier_metrics 
            else:
                raise Exception('not supported.')
            for m in _preload_metrics:
                one_result.append(_get_evaluated_result(y_pred, y_pred_proba, y_test, m))
        else:
            raise Exception('not supported.')
        result['metrics'].append(one_result)

    return result
