import akebono.features as features
from akebono.logging import getLogger
from akebono.io.dataset import load_dataset
from akebono.io.operation.dumper import dump_operation_result
from akebono.io.operation.loader import get_train_result 
from akebono.models import get_model
from akebono.utils import load_object_by_str
import os


logger = getLogger(__name__)


def train(operation_index, scenario_tag,
    dataset_config=None,
    model_config=None,
    feature_func='identify@akebono.features',
    feature_kwargs={},
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
            'feature_kwargs': feature_kwargs,
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
        fX = feature_func(X, **feature_kwargs)
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
            logger.info('dump_operation_result start.')
            dump_operation_result('train', operation_index, scenario_tag, ret)
            logger.info('dump_operation_result done.')


def predict(operation_index, scenario_tag,
    method_type='predict',
    dataset_config=None,
    model_config={},
    predicted_result_config={}
    ):
        if dataset_config is None:
            raise ValueError('dataset_config must be set.')
        if 'operation_index' not in dataset_config:
            dataset_config['operation_index'] = 0
        if 'scenario_tag' not in dataset_config:
            dataset_config['scenario_tag'] = 'latest'
        p_index = dataset_config['operation_index']
        p_tag = dataset_config['scenario_tag']

        tr = get_train_result(scenario_tag=p_tag, operation_index=p_index)
        if tr is None:
            raise Exception('target result not found.')

        if 'load_func_kwargs' not in dataset_config:
            dataset_config['load_func_kwargs'] = {}
        dataset_config['load_func_kwargs']['target_column'] = None # target_columnがNoneだと、predict用のDatasetが返ってくる
        dataset = load_dataset(dataset_config)
        feature_func = load_object_by_str(tr['feature_func'])
        X = dataset.value
        fX = feature_func(X, **tr['feature_kwargs'])
        
        model_config.update(tr['model_config'])
        model_config['is_rebuild'] = True
        model = get_model(model_config)

        predict_func = getattr(model, method_type, None)
        if predict_func is None:
            raise Exception('{} is not defined.'.format(method_type))
        print(predict_func(fX))
