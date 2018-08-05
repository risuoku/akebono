import akebono.features as features
from akebono.logging import getLogger
import akebono.io.dataset.bigquery as bq
from akebono.models import get as get_model
from akebono.utils import load_object_by_str
import os


logger = getLogger(__name__)


def train(
    model_kind=None,
    data_name=None, 
    model_name=None,
    load_dataset_kwargs={},
    feature_func='identify@akebono.features',
    feature_kwargs={},
    init_kwargs={},
    fit_kwargs={},
    evaluate_kwargs={},
    params={}
    ):
        if model_kind is None:
            raise ValueError('model_kind must be set.')
        dataset = bq.load(data_name, **load_dataset_kwargs)
        
        feature_func = load_object_by_str(feature_func)
        logger.debug('load dataset start.')
        X, y = dataset.get_predictor_target()
        logger.debug('load dataset done.')
        logger.debug('load feature start.')
        fX = feature_func(X, **feature_kwargs)
        logger.debug('load feature done.')
        
        model = get_model(model_kind, init_kwargs=init_kwargs, is_rebuild=False)
        
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
            model_name = model_name or data_name
            model.export(model_name)
            logger.info('save model done.')
