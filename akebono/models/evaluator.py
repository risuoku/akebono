import sklearn.metrics as skl_metrics
from akebono.utils import (
    load_object_by_str,
    get_label_by_object,
)


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


def _validate_metrics(o):
    if not isinstance(o, dict):
        raise TypeError('invalid type.')
    if ('name' not in o) or ('func' not in o) or ('predict_type' not in o):
        raise Exception('invalid key.')
    if o['predict_type'] not in ('predict', 'predict_proba',):
        raise Exception('invalid predict_type')


def _fit_and_predict(X_train, X_test, y_train, model, fit_kwargs):
    model.reset()
    model.fit(X_train, y_train, fit_kwargs)
    y_pred = y_pred_proba = None

    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    return model, y_pred, y_pred_proba


def _get_evaluated_result(y_pred, y_pred_proba, y_test, pos_index, metrics):
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
            if pos_index is None:
                raise Exception('pos_index must be set .. if predict_proba metric exists')
            y_pred_proba = y_pred_proba[:, pos_index]
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
    X, y,
    fit_kwargs,
    train_test_split_func='train_test_split@sklearn.model_selection',
    train_test_split_func_kwargs={},
    cross_val_iterator=None,
    cross_val_iterator_kwargs={},
    metrics='all',
    pos_index=None
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

    for X_train, X_test, y_train, y_test in train_test_iterator:
        one_result = []
        model, y_pred, y_pred_proba = _fit_and_predict(X_train, X_test, y_train, model, fit_kwargs)
        if metrics == 'all':
            if model_type == 'binary_classifier':
                for m in _binary_classifier_metrics:
                    one_result.append(_get_evaluated_result(y_pred, y_pred_proba, y_test, pos_index, m))
            else:
                raise Exception('not supported.')
        else:
            raise Exception('not supported.')
        result['metrics'].append(one_result)

    return result
