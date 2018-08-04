from akebono.utils import (
    load_object_by_str,
    get_label_by_object,
)
from akebono.logging import getLogger
import re


logger = getLogger(__name__)


class Metric:
    def __init__(self, func, label=None):
        self._func = func
        self._label = label
        if self._label is None:
            if re.search('sklearn.metrics.*', func.__module__) is not None:
                if func.__name__ in (
                    'accuracy_score', 'precision_score',
                    'recall_score', 'f1_score',
                ):
                    self._label = 'predict'
                elif func.__name__ in ('roc_curve',):
                    self._label = 'predict_proba'
                else:
                    raise Exception('unsupported autodetected metrics')
            else:
                raise Exception('unsupported metrics module')
        else:
            if self._label not in ('predict', 'predict_proba'):
                raise ValueError('invalid label')
    
    def get_func_and_label(self):
        return self._func, self._label


def evaluate(X, y, model, fit_kwargs,
    train_test_split_func='train_test_split@sklearn.model_selection', train_test_split_func_kwargs={},
    cross_val_iterator=None, cross_val_iterator_kwargs={},
    metric_func='accuracy_score@sklearn.metrics', metric_func_kwargs={},
    pos_index=None,
    metric_cls=Metric):
    
    # cross_val_iteratorはNone or iterable
    if cross_val_iterator is not None:
        if isinstance(cross_val_iterator, str):
            cross_val_iterator = load_object_by_str(cross_val_iterator)
    else:
        # train_test_split_func はNone or string or function
        if isinstance(train_test_split_func, str):
            train_test_split_func = load_object_by_str(train_test_split_func)
    # metric_funcはstring or [string] or function or [function]
    if not isinstance(metric_func, list) and not isinstance(metric_func_kwargs, list):
        metric_func = [metric_func]
        metric_func_kwargs = [metric_func_kwargs]
    if isinstance(metric_func, list) and isinstance(metric_func_kwargs, list):
        metric_list = list(zip(
            [metric_cls(load_object_by_str(mf)) if isinstance(mf, str) else mf for mf in metric_func],
            metric_func_kwargs
        ))
    else:
        raise TypeError('invalid type')
    
    # cross_valがNoneでなかったらCVモード。test_split_funcは無視される。
    report = {}
    if cross_val_iterator is None:
        report['cross_val_enabled'] = False
        report['metrics'] = []
        logger.debug('cross_val_iterator is None')
        X_train, X_test, y_train, y_test = train_test_split_func(X, y, random_state=0)
        model.reset()
        model.fit(X_train, y_train, fit_kwargs)
        for mt, mfkwargs in metric_list:
            mf, mflabel = mt.get_func_and_label()
            if mflabel == 'predict':
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict_proba(X_test)
                if pos_index is None:
                    raise Exception('pos_index must be set .. if predict_proba metric exists')
                y_pred = y_pred[:, pos_index]
            r = mf(y_test, y_pred, **mfkwargs)
            report['metrics'].append({
                'name': get_label_by_object(mf),
                'result': r
            })
    else:
        logger.debug('cross_val_iterator is valid')
        report['cross_val_enabled'] = True
        report['attempts'] = []
        for train_index, test_index in cross_val_iterator(**cross_val_iterator_kwargs).split(X.index):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            mets = {'metrics': []}
            model.reset()
            model.fit(X_train, y_train, fit_kwargs)
            for mt, mfkwargs in metric_list:
                mf, mflabel = mt.get_func_and_label()
                if mflabel == 'predict':
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict_proba(X_test)
                    if pos_index is None:
                        raise Exception('pos_index must be set .. if predict_proba metric exists')
                    y_pred = y_pred[:, pos_index]
                r = mf(y_test, y_pred, **mfkwargs)
                mets['metrics'].append({
                    'name': get_label_by_object(mf),
                    'result': r
                })
            report['attempts'].append(mets)
    return report

