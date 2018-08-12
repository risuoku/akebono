from akebono.utils import (
    load_object_by_str,
    cache_located_at,
    pathjoin,
    Param,
)
import akebono.settings as settings
from .model import Dataset
from akebono.logging import getLogger
import copy


logger = getLogger(__name__)


def get_dataset(dataset_config):
    dataset_name = dataset_config.get('name')
    target_column = dataset_config.get('target_column', 'target')
    cache_enabled = dataset_config.get('cache_enabled', False)
    loader_config = dataset_config.get('loader_config')
    if not isinstance(loader_config, dict):
        raise Exception('loader_config must be specified and this type is dict.')
    load_func = loader_config.get('func')
    if load_func is None:
        raise Exception('loader_config.func must be specified.')
    load_func = load_object_by_str(load_func)
    load_func_kwargs = Param(loader_config.get('func_kwargs', {}))
    loader_param = loader_config.get('param', {})
    _reserved_params = ('dataset_name', 'target_column',)
    for rp in _reserved_params:
        if rp in loader_param:
            raise KeyError('{} is reserved param.'.format(rp))
    loader_param['dataset_name'] = dataset_name
    loader_param['target_column'] = target_column

    preprocess_func = load_object_by_str(dataset_config.get('preprocess_func', 'identify@akebono.dataset.preprocessors'))
    preprocess_func_kwargs = Param(dataset_config.get('preprocess_func_kwargs', {}))

    def _core_func():
        return preprocess_func(
            load_func(copy.copy(load_func_kwargs.value), loader_param),
            **copy.copy(preprocess_func_kwargs.value)
        )

    if cache_enabled:
        if dataset_name is not None:
            logger.info('dataset cache enabled')
            fname = '{}_{}{}.pkl'.format(
                dataset_name,
                load_func_kwargs.get_hashed_id(length=32),
                preprocess_func_kwargs.get_hashed_id(length=32)
            )
            _core_func = cache_located_at(pathjoin(settings.cache_dir, fname))(_core_func)
        else:
            raise Exception('dataset_config.cache_enabled is True, but dataset_config.name is None')

    return Dataset(_core_func(), target_column)
