import akebono.features as features
from akebono.logging import getLogger
from akebono.io.dataset import load_dataset
from akebono.io.operation.dumper import (
    dump_train_result,
    dump_predicted_result,
)
from akebono.io.operation.loader import get_train_result 
from akebono.models import get_model
from akebono.utils import load_object_by_str
import akebono.settings as settings
import os


logger = getLogger(__name__)


def train(operation_index, scenario_tag,
    dataset_config=None,
    model_config=None,
    feature_func='identify@akebono.features',
    feature_func_kwargs={},
    evaluate_enabled=False,
    fit_model_enabled=False,
    dump_result_enabled=False
    ):
        if model_config is None:
            raise ValueError('model_config must be set.')
        if dataset_config is None:
            raise ValueError('dataset_config must be set.')

        ret = {
            'type': 'train',
            'index': operation_index,
            'dataset_config': dataset_config,
            'model_config': model_config,
            'feature_func': feature_func,
            'feature_func_kwargs': feature_func_kwargs,
            'evaluate_enabled': evaluate_enabled,
            'fit_model_enabled': fit_model_enabled,
            'dump_result_enabled': dump_result_enabled
        }

        dataset = load_dataset(dataset_config)
        
        feature_func = load_object_by_str(feature_func)
        logger.debug('load dataset start.')
        X, y = dataset.get_predictor_target()
        logger.debug('load dataset done.')
        logger.debug('load feature start.')
        fX = feature_func(X, **feature_func_kwargs)
        logger.debug('load feature done.')
        
        model_config['is_rebuild'] = False
        model = get_model(model_config)
        
        if evaluate_enabled:
            logger.debug('evaluate start.')
            rep = model.evaluate(fX, y)
            logger.debug('evaluate done.')
            ret['evaluate'] = rep
        if fit_model_enabled:
            logger.info('fit start.')
            model.fit(fX, y)
            logger.info('fit done.')
            ret['model'] = model
        if dump_result_enabled:
            logger.info('dump_train_result start.')
            dump_train_result(operation_index, scenario_tag, ret)
            logger.info('dump_train_result done.')
        
        return ret


def predict(operation_index, scenario_tag,
    method_type='predict',
    dataset_config=None,
    model_config={},
    dump_result_enabled=False,
    dump_result_format='csv',
    result_target_columns='all',
    result_predict_column='predicted'
    ):
        if dataset_config is None:
            raise ValueError('dataset_config must be set.')

        ret = {
            'type': 'predict',
            'method_type': method_type,
            'dataset_config': dataset_config,
            'model_config': model_config,
            'dump_result_enabled': dump_result_enabled,
            'dump_result_format': dump_result_format,
            'result_target_columns': result_target_columns,
            'result_predict_column': result_predict_column,
        }

        if 'operation_index' not in model_config:
            model_config['operation_index'] = 0
        model_config['scenario_tag'] = scenario_tag
        p_index = model_config['operation_index']

        tr = get_train_result(scenario_tag=scenario_tag, operation_index=p_index)
        if tr is None:
            raise Exception('target result not found.')
        ret['train_result'] = tr

        if 'load_func_kwargs' not in dataset_config:
            dataset_config['load_func_kwargs'] = {}
        dataset_config['load_func_kwargs']['target_column'] = None # target_columnがNoneだと、predict用のDatasetが返ってくる
        dataset = load_dataset(dataset_config)
        feature_func = load_object_by_str(tr['feature_func'])
        X = dataset.value
        fX = feature_func(X, **tr['feature_func_kwargs'])
        
        model_config.update(tr['model_config'])
        model_config['is_rebuild'] = True
        model = get_model(model_config)

        predict_func = getattr(model, method_type, None)
        if predict_func is None:
            raise Exception('{} is not defined.'.format(method_type))
        rawresult = predict_func(fX)
        predict_result = fX
        if not result_target_columns == 'all':
            if not isinstance(result_target_columns, list):
                raise TypeError('result_target_columns must be list.')
            predict_result = fX[result_target_columns]
        predict_result[result_predict_column] = rawresult

        if dump_result_enabled:
            logger.debug('dump_predicted_result start.')
            dump_predicted_result(operation_index, scenario_tag, dump_result_format, predict_result, ret)
            logger.debug('dump_predicted_result done.')

        ret['predict_result'] = predict_result
        return ret
