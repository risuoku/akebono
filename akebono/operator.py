import akebono.features as features
from akebono.logging import getLogger
from akebono.io.dataset import load_dataset
from akebono.models import get as get_model
from akebono.utils import load_object_by_str
import os


logger = getLogger(__name__)


def train(operation_index,
    dataset_config=None,
    model_name=None,
    feature_func='identify@akebono.features',
    feature_kwargs={},
    init_kwargs={},
    fit_kwargs={},
    evaluate_kwargs={},
    params={}
    ):
        if model_name is None:
            raise ValueError('model_name must be set.')
        if dataset_config is None:
            raise ValueError('dataset_config must be set.')
        dataset = load_dataset(dataset_config)
        
        feature_func = load_object_by_str(feature_func)
        logger.debug('load dataset start.')
        X, y = dataset.get_predictor_target()
        logger.debug('load dataset done.')
        logger.debug('load feature start.')
        fX = feature_func(X, **feature_kwargs)
        logger.debug('load feature done.')
        
        model = get_model(model_name, init_kwargs=init_kwargs, is_rebuild=False)
        
        ret = {
            'type': 'train',
            'index': operation_index,
            'dataset_config': dataset_config,
            'model_name': model_name,
        }
        if params.get('evaluate_enabled') is True:
            logger.debug('evaluate start.')
            rep = model.evaluate(fX, y, fit_kwargs, **evaluate_kwargs)
            print(rep)
            logger.debug('evaluate end.')
        if params.get('save_enabled') is True:
            logger.info('save model start.')
            logger.info('fit start.')
            model.fit(fX, y, fit_kwargs)
            logger.info('fit done.')
            model.dump(dataset_config['name'])
            logger.info('save model done.')
