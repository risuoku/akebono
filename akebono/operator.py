import akebono.features as features
from akebono.logging import getLogger
from akebono.io.dataset import load_dataset
from akebono.models import get_model
from akebono.utils import load_object_by_str
import os


logger = getLogger(__name__)


def train(operation_index,
    dataset_config=None,
    model_config=None,
    feature_func='identify@akebono.features',
    feature_kwargs={},
    save_enabled=False,
    evaluate_enabled=False
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
        }

        dataset = load_dataset(dataset_config)
        
        feature_func = load_object_by_str(feature_func)
        logger.debug('load dataset start.')
        X, y = dataset.get_predictor_target()
        logger.debug('load dataset done.')
        logger.debug('load feature start.')
        fX = feature_func(X, **feature_kwargs)
        logger.debug('load feature done.')
        
        model = get_model(model_config)
        
        if evaluate_enabled:
            logger.debug('evaluate start.')
            rep = model.evaluate(fX, y)
            print(rep)
            logger.debug('evaluate end.')
        if save_enabled:
            logger.info('save model start.')
            logger.info('fit start.')
            model.fit(fX, y)
            logger.info('fit done.')
            model.dump(dataset_config['name'])
            logger.info('save model done.')
